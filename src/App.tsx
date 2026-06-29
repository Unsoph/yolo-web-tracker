import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import { processYoloOutput } from './utils/yoloUtils';
import { SimpleTracker, type TrackedObject } from './utils/tracker';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [fps, setFps] = useState(0);

  // Refs for the animation loop (these persist without causing re-renders)
  const trackerRef = useRef(new SimpleTracker());
  const selectedTrackIdRef = useRef<number | null>(null);
  const currentTracksRef = useRef<TrackedObject[]>([]);
  const facingModeRef = useRef<'user' | 'environment'>('environment');
  const runningRef = useRef(true);

  // Keep ref in sync with state
  useEffect(() => {
    selectedTrackIdRef.current = selectedTrackId;
  }, [selectedTrackId]);

  // ---- Hit-test logic shared between mouse and touch ----
  const hitTest = useCallback((clientX: number, clientY: number) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;

    // Pad the hitbox by 15px on each side so tapping on mobile is forgiving
    const pad = 15;
    let clickedOnObject = false;

    for (const track of currentTracksRef.current) {
      if (
        x >= track.x1 - pad && x <= track.x2 + pad &&
        y >= track.y1 - pad && y <= track.y2 + pad
      ) {
        setSelectedTrackId(track.trackId);
        clickedOnObject = true;
        break;
      }
    }

    if (!clickedOnObject) {
      setSelectedTrackId(null);
    }
  }, []);

  // Mouse click handler (desktop)
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    hitTest(e.clientX, e.clientY);
  }, [hitTest]);

  // Touch handler (mobile)
  const handleCanvasTouch = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // prevent double-firing with click
    if (e.touches.length > 0) {
      hitTest(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, [hitTest]);

  // Function to switch between front and back camera
  const toggleCamera = async () => {
    const newMode = facingModeRef.current === 'user' ? 'environment' : 'user';
    facingModeRef.current = newMode;

    // Stop existing stream
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: newMode },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => {
          if (videoRef.current) videoRef.current.onloadedmetadata = resolve;
        });
        videoRef.current.play();
      }
    } catch (err) {
      console.error("Camera switch error:", err);
    }
  };

  // ---- Main effect: camera + model + inference loop ----
  useEffect(() => {
    let session: ort.InferenceSession;
    runningRef.current = true;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: facingModeRef.current },
          audio: false
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await new Promise((resolve) => {
            if (videoRef.current) videoRef.current.onloadedmetadata = resolve;
          });
          videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera error:", err);
      }
    };

    const loadModelAndRun = async () => {
      await setupCamera();

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

      // ---- GATED INFERENCE LOOP ----
      // Instead of requestAnimationFrame (which fires before inference finishes),
      // we use a while-loop with awaits. The next frame only starts AFTER the
      // previous inference is completely done. This prevents frame pile-up.
      let lastTime = performance.now();

      while (runningRef.current) {
        if (!videoRef.current || !canvasRef.current || !offscreenCtx || !session) {
          await new Promise(r => setTimeout(r, 100));
          continue;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) break;

        if (video.videoWidth === 0 || video.videoHeight === 0) {
          await new Promise(r => setTimeout(r, 100));
          continue;
        }

        // Match canvas to video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        // Draw video → 640x640 offscreen canvas for YOLO
        offscreenCtx.drawImage(video, 0, 0, 640, 640);
        const imgData = offscreenCtx.getImageData(0, 0, 640, 640).data;

        // RGBA → CHW float32 normalized
        const float32Data = new Float32Array(3 * 640 * 640);
        for (let i = 0; i < 640 * 640; i++) {
          float32Data[i]                   = imgData[i * 4]     / 255.0; // R
          float32Data[640 * 640 + i]       = imgData[i * 4 + 1] / 255.0; // G
          float32Data[2 * 640 * 640 + i]   = imgData[i * 4 + 2] / 255.0; // B
        }

        // Run inference (this is the slow part — we AWAIT it)
        const tensor = new ort.Tensor('float32', float32Data, inputShape);
        const results = await session.run({ images: tensor });

        const outputTensor = results.output0;
        const boxes = processYoloOutput(outputTensor.data as Float32Array);

        // Scale boxes from 640x640 back to video resolution
        const scaleX = video.videoWidth / 640;
        const scaleY = video.videoHeight / 640;
        for (const box of boxes) {
          box.x1 *= scaleX;
          box.x2 *= scaleX;
          box.y1 *= scaleY;
          box.y2 *= scaleY;
        }

        // Run tracker
        const tracks = trackerRef.current.update(boxes);
        currentTracksRef.current = tracks;

        // Draw bounding boxes on canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (const track of tracks) {
          if (selectedTrackIdRef.current !== null && track.trackId !== selectedTrackIdRef.current) {
            continue;
          }

          const isLocked = track.trackId === selectedTrackIdRef.current;
          const color = isLocked ? '#ff0000' : '#00ff00';
          const lineW = isLocked ? 4 : 2;

          ctx.strokeStyle = color;
          ctx.lineWidth = lineW;
          ctx.beginPath();
          ctx.rect(track.x1, track.y1, track.x2 - track.x1, track.y2 - track.y1);
          ctx.stroke();

          // Label with background for readability
          const label = isLocked ? `LOCKED ID: ${track.trackId}` : `ID: ${track.trackId}`;
          ctx.font = 'bold 14px Arial';
          const textW = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0,0,0,0.6)';
          ctx.fillRect(track.x1, track.y1 - 20, textW + 8, 20);
          ctx.fillStyle = color;
          ctx.fillText(label, track.x1 + 4, track.y1 - 5);
        }

        // FPS counter
        const now = performance.now();
        const delta = now - lastTime;
        lastTime = now;
        if (delta > 0) setFps(Math.round(1000 / delta));

        // Yield to the browser so the UI stays responsive
        await new Promise(r => requestAnimationFrame(r));
      }
    };

    loadModelAndRun();

    return () => {
      runningRef.current = false;
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: '10px', background: '#111', color: '#eee', minHeight: '100vh' }}>
      <h2 style={{ margin: '5px 0' }}>Drone Tracker</h2>
      <p style={{ fontSize: '13px', margin: '5px 0 10px', color: '#aaa' }}>
        {modelLoaded
          ? "Tap a bounding box to lock. Tap background to unlock."
          : "Loading YOLOv8 model…"}
      </p>

      {/* Controls */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '10px', flexWrap: 'wrap' }}>
        <button
          onClick={toggleCamera}
          style={{
            padding: '10px 20px',
            fontSize: '15px',
            backgroundColor: '#0070f3',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer'
          }}
        >
          Flip Camera 🔄
        </button>
        {selectedTrackId !== null && (
          <button
            onClick={() => setSelectedTrackId(null)}
            style={{
              padding: '10px 20px',
              fontSize: '15px',
              backgroundColor: '#e00',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
          >
            Unlock Target 🔓
          </button>
        )}
      </div>

      {selectedTrackId !== null && (
        <h3 style={{ color: '#ff4444', margin: '5px 0' }}>🔒 Target Locked: ID {selectedTrackId}</h3>
      )}

      {modelLoaded && (
        <p style={{ fontSize: '12px', color: '#888', margin: '2px 0 8px' }}>FPS: {fps}</p>
      )}

      <div style={{ position: 'relative', display: 'inline-block', width: '100%', maxWidth: '800px' }}>
        <video
          ref={videoRef}
          style={{
            display: 'block',
            borderRadius: '8px',
            border: '2px solid #333',
            width: '100%',
            height: 'auto'
          }}
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onTouchStart={handleCanvasTouch}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            cursor: 'crosshair',
            width: '100%',
            height: '100%',
            touchAction: 'none'   // prevents browser from hijacking touch gestures
          }}
        />
      </div>
    </div>
  );
}

export default App;
