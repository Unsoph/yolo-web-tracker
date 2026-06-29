import React, { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import { processYoloOutput } from './utils/yoloUtils';
import { SimpleTracker, type TrackedObject } from './utils/tracker';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  // We keep these in refs so they persist across renders and the animation loop can access them
  const trackerRef = useRef(new SimpleTracker());
  const selectedTrackIdRef = useRef<number | null>(null);
  const currentTracksRef = useRef<TrackedObject[]>([]);
  const facingModeRef = useRef<'user' | 'environment'>('environment');

  // Update ref when state changes
  useEffect(() => {
    selectedTrackIdRef.current = selectedTrackId;
  }, [selectedTrackId]);

  // Function to switch between front and back camera
  const toggleCamera = async () => {
    const newMode = facingModeRef.current === 'user' ? 'environment' : 'user';
    facingModeRef.current = newMode;
    
    // Stop existing stream
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }

    // Start new stream with new facingMode
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: newMode }, 
        audio: false 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = resolve;
          }
        });
        videoRef.current.play();
      }
    } catch (err) {
      console.error("Camera switch error:", err);
    }
  };

  useEffect(() => {
    let session: ort.InferenceSession;
    let animationId: number;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: facingModeRef.current }, 
          audio: false 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Wait for video to be ready
          await new Promise((resolve) => {
            if (videoRef.current) {
              videoRef.current.onloadedmetadata = resolve;
            }
          });
          videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera error:", err);
      }
    };

    const loadModelAndRun = async () => {
      await setupCamera();

      // Load ONNX Model
      try {
        session = await ort.InferenceSession.create('/yolov8n.onnx');
        setModelLoaded(true);
      } catch (e) {
        console.error("Failed to load ONNX model:", e);
        return;
      }

      const inputShape = [1, 3, 640, 640];
      const offscreenCanvas = document.createElement('canvas');
      offscreenCanvas.width = 640;
      offscreenCanvas.height = 640;
      const offscreenCtx = offscreenCanvas.getContext('2d');

      const processFrame = async () => {
        if (!videoRef.current || !canvasRef.current || !offscreenCtx || !session) {
          animationId = requestAnimationFrame(processFrame);
          return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Make sure video is ready
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            animationId = requestAnimationFrame(processFrame);
            return;
        }

        // Match canvas size to video size
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        // Draw video to offscreen canvas (scaling it to 640x640 for YOLO)
        offscreenCtx.drawImage(video, 0, 0, 640, 640);
        const imgData = offscreenCtx.getImageData(0, 0, 640, 640).data;

        // Convert ImageData (RGBA) to Float32Array RGB [1, 3, 640, 640] normalized to 0-1
        const float32Data = new Float32Array(3 * 640 * 640);
        for (let i = 0; i < 640 * 640; i++) {
            float32Data[i] = imgData[i * 4] / 255.0; // R
            float32Data[640 * 640 + i] = imgData[i * 4 + 1] / 255.0; // G
            float32Data[2 * 640 * 640 + i] = imgData[i * 4 + 2] / 255.0; // B
        }

        // Run Inference
        const tensor = new ort.Tensor('float32', float32Data, inputShape);
        const results = await session.run({ images: tensor });
        
        // Output from YOLOv8 is named 'output0'
        const outputTensor = results.output0; 
        const boxes = processYoloOutput(outputTensor.data as Float32Array);

        // Map boxes from 640x640 back to the original video dimensions
        const scaleX = video.videoWidth / 640;
        const scaleY = video.videoHeight / 640;
        boxes.forEach(box => {
            box.x1 *= scaleX;
            box.x2 *= scaleX;
            box.y1 *= scaleY;
            box.y2 *= scaleY;
        });

        // Run Tracker
        const tracks = trackerRef.current.update(boxes);
        currentTracksRef.current = tracks;

        // Draw on main canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        for (const track of tracks) {
            // Only draw if we haven't selected something, OR if this is the selected thing
            if (selectedTrackIdRef.current !== null && track.trackId !== selectedTrackIdRef.current) {
                continue;
            }

            const isLocked = track.trackId === selectedTrackIdRef.current;
            ctx.strokeStyle = isLocked ? '#ff0000' : '#00ff00';
            ctx.lineWidth = isLocked ? 4 : 2;
            
            ctx.beginPath();
            ctx.rect(track.x1, track.y1, track.x2 - track.x1, track.y2 - track.y1);
            ctx.stroke();

            // Draw label
            ctx.fillStyle = isLocked ? '#ff0000' : '#00ff00';
            ctx.font = '16px Arial';
            ctx.fillText(
                isLocked ? `LOCKED ID: ${track.trackId}` : `ID: ${track.trackId}`, 
                track.x1, 
                track.y1 - 5
            );
        }

        // Loop
        animationId = requestAnimationFrame(processFrame);
      };

      processFrame();
    };

    loadModelAndRun();

    return () => {
      cancelAnimationFrame(animationId);
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Handle canvas clicks for target locking
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    
    // Get click coordinates relative to the canvas
    const rect = canvasRef.current.getBoundingClientRect();
    
    // Scale coordinates if the canvas is responsive on mobile
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    let clickedOnObject = false;

    // Check if the click falls inside any tracked object
    for (const track of currentTracksRef.current) {
        if (x >= track.x1 && x <= track.x2 && y >= track.y1 && y <= track.y2) {
            setSelectedTrackId(track.trackId);
            clickedOnObject = true;
            break;
        }
    }

    if (!clickedOnObject) {
        setSelectedTrackId(null);
    }
  };

  return (
    <div style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: '10px' }}>
      <h2>Drone Tracker (Web)</h2>
      <p style={{ fontSize: '14px', marginBottom: '10px' }}>
        {modelLoaded 
            ? "Model loaded! Click on a bounding box to lock onto it. Click the background to clear." 
            : "Loading YOLOv8 ONNX Model (this may take a few seconds)..."}
      </p>

      {/* Camera Toggle Button */}
      <button 
        onClick={toggleCamera} 
        style={{ 
          marginBottom: '15px', 
          padding: '10px 20px', 
          fontSize: '16px',
          backgroundColor: '#0070f3',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer'
        }}
      >
        Flip Camera 🔄
      </button>

      {selectedTrackId !== null && (
        <h3 style={{ color: 'red', marginTop: '0' }}>Target Locked: ID {selectedTrackId}</h3>
      )}

      <div style={{ position: 'relative', display: 'inline-block', width: '100%', maxWidth: '800px' }}>
        <video 
            ref={videoRef} 
            style={{ 
              display: 'block', 
              borderRadius: '8px', 
              border: '2px solid #ccc',
              width: '100%',
              height: 'auto'
            }} 
            playsInline
            muted
        />
        <canvas 
            ref={canvasRef} 
            onClick={handleCanvasClick}
            style={{ 
                position: 'absolute', 
                top: 0, 
                left: 0, 
                cursor: 'crosshair',
                width: '100%',
                height: '100%'
            }} 
        />
      </div>
    </div>
  );
}

export default App;
