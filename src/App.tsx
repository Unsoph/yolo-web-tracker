import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import { processYoloOutput, type BoundingBox } from './utils/yoloUtils';
import { SimpleTracker, type TrackedObject } from './utils/tracker';

// ---- CONSTANTS ----
const MODEL_INPUT_SIZE = 320;  // Re-exported ONNX model at 320x320 for ~4x speedup
const PIXELS = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

/** Helper to extract a 3x3 color patch from the float32 CHW image data */
function extractColorPatch(box: BoundingBox, float32Data: Float32Array): number[] {
  const patch: number[] = [];
  const stepX = (box.x2 - box.x1) / 3;
  const stepY = (box.y2 - box.y1) / 3;
  
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      let cx = Math.floor(box.x1 + stepX * (col + 0.5));
      let cy = Math.floor(box.y1 + stepY * (row + 0.5));
      
      cx = Math.max(0, Math.min(MODEL_INPUT_SIZE - 1, cx));
      cy = Math.max(0, Math.min(MODEL_INPUT_SIZE - 1, cy));
      
      const idx = cy * MODEL_INPUT_SIZE + cx;
      const r = float32Data[idx];
      const g = float32Data[PIXELS + idx];
      const b = float32Data[2 * PIXELS + idx];
      
      patch.push(r, g, b);
    }
  }
  return patch;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [fps, setFps] = useState(0);

  const trackerRef = useRef(new SimpleTracker());
  const selectedTrackIdRef = useRef<number | null>(null);
  const currentTracksRef = useRef<TrackedObject[]>([]);
  const facingModeRef = useRef<'user' | 'environment'>('environment');
  const runningRef = useRef(true);

  useEffect(() => {
    selectedTrackIdRef.current = selectedTrackId;
  }, [selectedTrackId]);

  // ---- Shared hit-test for mouse + touch ----
  const hitTest = useCallback((clientX: number, clientY: number) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;

    const pad = 20;
    
    // Z-INDEX FIX: Sort tracks by area ascending (smallest first)
    // This way, if a smaller box is completely inside a larger box, it gets clicked first.
    const sortedTracks = [...currentTracksRef.current].sort((a, b) => {
      const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
      const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
      return areaA - areaB;
    });

    let clickedOnObject = false;

    for (const track of sortedTracks) {
      if (
        x >= track.x1 - pad && x <= track.x2 + pad &&
        y >= track.y1 - pad && y <= track.y2 + pad
      ) {
        setSelectedTrackId(track.trackId);
        clickedOnObject = true;
        break;
      }
    }
    if (!clickedOnObject) setSelectedTrackId(null);
  }, []);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    hitTest(e.clientX, e.clientY);
  }, [hitTest]);

  const handleCanvasTouch = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (e.touches.length > 0) hitTest(e.touches[0].clientX, e.touches[0].clientY);
  }, [hitTest]);

  const toggleCamera = async () => {
    const newMode = facingModeRef.current === 'user' ? 'environment' : 'user';
    facingModeRef.current = newMode;

    if (videoRef.current && videoRef.current.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: newMode },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise(r => { if (videoRef.current) videoRef.current.onloadedmetadata = r; });
        videoRef.current.play();
      }
    } catch (err) {
      console.error("Camera switch error:", err);
    }
  };

  // ---- Main effect ----
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
          await new Promise(r => { if (videoRef.current) videoRef.current.onloadedmetadata = r; });
          videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera error:", err);
      }
    };

    const run = async () => {
      await setupCamera();

      try {
        session = await ort.InferenceSession.create('/yolov8n.onnx');
        setModelLoaded(true);
      } catch (e) {
        console.error("Failed to load ONNX model:", e);
        return;
      }

      // ---- PRE-ALLOCATE BUFFERS (no GC pressure per frame) ----
      const inputShape = [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE];
      const float32Data = new Float32Array(3 * PIXELS); // reused every frame

      const offscreen = document.createElement('canvas');
      offscreen.width = MODEL_INPUT_SIZE;
      offscreen.height = MODEL_INPUT_SIZE;
      const offCtx = offscreen.getContext('2d', { willReadFrequently: true })!;

      let lastTime = performance.now();
      let frameCount = 0;

      // ---- GATED INFERENCE LOOP ----
      while (runningRef.current) {
        if (!videoRef.current || !canvasRef.current || !session) {
          await new Promise(r => setTimeout(r, 50));
          continue;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx || video.videoWidth === 0) {
          await new Promise(r => setTimeout(r, 50));
          continue;
        }

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        // Draw video → small offscreen canvas
        offCtx.drawImage(video, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
        const imgData = offCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;

        // RGBA → CHW float32 (reusing pre-allocated buffer)
        for (let i = 0; i < PIXELS; i++) {
          const base = i * 4;
          float32Data[i]             = imgData[base]     / 255.0;
          float32Data[PIXELS + i]    = imgData[base + 1] / 255.0;
          float32Data[2 * PIXELS + i] = imgData[base + 2] / 255.0;
        }

        // Run inference
        const tensor = new ort.Tensor('float32', float32Data, inputShape);
        const results = await session.run({ images: tensor });
        const output = results.output0;
        const outputData = output.data as Float32Array;

        // Get number of anchors from output shape
        const numAnchors = output.dims[2];
        const boxes = processYoloOutput(outputData, 0.45, 0.3, numAnchors);

        // ROBUST LOCK: Extract color features before scaling boxes
        for (const box of boxes) {
          box.colorPatch = extractColorPatch(box, float32Data);
        }

        // Scale boxes back to video resolution
        const scaleX = video.videoWidth / MODEL_INPUT_SIZE;
        const scaleY = video.videoHeight / MODEL_INPUT_SIZE;
        for (const box of boxes) {
          box.x1 *= scaleX;
          box.x2 *= scaleX;
          box.y1 *= scaleY;
          box.y2 *= scaleY;
        }

        // Track
        const tracks = trackerRef.current.update(boxes, selectedTrackIdRef.current);
        currentTracksRef.current = tracks;

        // Draw
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Z-INDEX FIX: Sort tracks by area descending (largest first)
        // This way, largest boxes are drawn first, and smaller ones are drawn ON TOP of them.
        const sortedTracksToDraw = [...tracks].sort((a, b) => {
          const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
          const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
          return areaB - areaA;
        });

        // Ensure the locked track is always drawn absolute last (on top of everything)
        const lockedTracks = sortedTracksToDraw.filter(t => t.trackId === selectedTrackIdRef.current);
        const otherTracks = sortedTracksToDraw.filter(t => t.trackId !== selectedTrackIdRef.current);
        const finalDrawOrder = [...otherTracks, ...lockedTracks];

        for (const track of finalDrawOrder) {
          // If we have a locked target, don't draw others
          if (selectedTrackIdRef.current !== null && track.trackId !== selectedTrackIdRef.current) {
            continue;
          }

          const isLocked = track.trackId === selectedTrackIdRef.current;
          const color = isLocked ? '#ff3333' : '#00ff88';
          const lineW = isLocked ? 4 : 2;

          ctx.strokeStyle = color;
          ctx.lineWidth = lineW;
          ctx.beginPath();
          ctx.rect(track.x1, track.y1, track.x2 - track.x1, track.y2 - track.y1);
          ctx.stroke();

          const label = isLocked ? `LOCKED #${track.trackId}` : `#${track.trackId}`;
          ctx.font = 'bold 14px Arial';
          const tw = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0,0,0,0.65)';
          ctx.fillRect(track.x1, track.y1 - 20, tw + 8, 20);
          ctx.fillStyle = color;
          ctx.fillText(label, track.x1 + 4, track.y1 - 5);
        }

        // FPS (smoothed over 10 frames)
        frameCount++;
        if (frameCount >= 10) {
          const now = performance.now();
          setFps(Math.round(10000 / (now - lastTime)));
          lastTime = now;
          frameCount = 0;
        }

        // Yield to browser
        await new Promise(r => requestAnimationFrame(r));
      }
    };

    run();

    return () => {
      runningRef.current = false;
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: '10px', background: '#111', color: '#eee', minHeight: '100vh' }}>
      <h2 style={{ margin: '5px 0' }}>Drone Tracker</h2>
      <p style={{ fontSize: '13px', margin: '5px 0 10px', color: '#aaa' }}>
        {modelLoaded
          ? "Tap a person to lock. Tap background to unlock."
          : "Loading YOLOv8 model…"}
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '10px', flexWrap: 'wrap' }}>
        <button onClick={toggleCamera} style={btnStyle('#0070f3')}>
          Flip Camera 🔄
        </button>
        {selectedTrackId !== null && (
          <button onClick={() => setSelectedTrackId(null)} style={btnStyle('#e00')}>
            Unlock 🔓
          </button>
        )}
      </div>

      {selectedTrackId !== null && (
        <h3 style={{ color: '#ff4444', margin: '5px 0' }}>🔒 Locked: #{selectedTrackId}</h3>
      )}

      {modelLoaded && (
        <p style={{ fontSize: '12px', color: '#888', margin: '2px 0 8px' }}>FPS: {fps}</p>
      )}

      <div style={{ position: 'relative', display: 'inline-block', width: '100%', maxWidth: '800px' }}>
        <video
          ref={videoRef}
          style={{ display: 'block', borderRadius: '8px', border: '2px solid #333', width: '100%', height: 'auto' }}
          playsInline muted
        />
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onTouchStart={handleCanvasTouch}
          style={{
            position: 'absolute', top: 0, left: 0,
            cursor: 'crosshair', width: '100%', height: '100%',
            touchAction: 'none'
          }}
        />
      </div>
    </div>
  );
}

const btnStyle = (bg: string): React.CSSProperties => ({
  padding: '10px 20px',
  fontSize: '15px',
  backgroundColor: bg,
  color: 'white',
  border: 'none',
  borderRadius: '8px',
  cursor: 'pointer',
});

export default App;
