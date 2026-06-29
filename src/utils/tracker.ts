import { type BoundingBox, calculateIoU } from "./yoloUtils";

export interface TrackedObject extends BoundingBox {
    trackId: number;
    missedFrames: number;
    // Velocity estimation for prediction
    vx: number;  // pixels per frame
    vy: number;
    // Last known position (for velocity calc)
    lastCx: number;
    lastCy: number;
}

/**
 * Enhanced IoU Tracker with:
 * - Velocity-based position prediction (so tracks survive temporary occlusion)
 * - Long memory (90 frames = ~3 seconds at 30fps before forgetting)
 * - Distance-based fallback matching for re-identification after leaving frame
 */
export class SimpleTracker {
    private tracks: TrackedObject[] = [];
    private nextId: number = 1;
    private maxMissedFrames: number = 90; // ~3 seconds at 30fps before giving up

    public update(detections: BoundingBox[]): TrackedObject[] {
        // First frame: initialise all tracks
        if (this.tracks.length === 0) {
            this.tracks = detections.map(det => ({
                ...det,
                trackId: this.nextId++,
                missedFrames: 0,
                vx: 0,
                vy: 0,
                lastCx: (det.x1 + det.x2) / 2,
                lastCy: (det.y1 + det.y2) / 2,
            }));
            return this.getVisibleTracks();
        }

        // For tracks that are currently missing, predict where they SHOULD be
        // using their last known velocity
        for (const track of this.tracks) {
            if (track.missedFrames > 0) {
                const w = track.x2 - track.x1;
                const h = track.y2 - track.y1;
                const cx = (track.x1 + track.x2) / 2 + track.vx;
                const cy = (track.y1 + track.y2) / 2 + track.vy;
                track.x1 = cx - w / 2;
                track.y1 = cy - h / 2;
                track.x2 = cx + w / 2;
                track.y2 = cy + h / 2;
            }
        }

        const usedDetections = new Set<number>();
        const matchedTracks = new Set<number>();

        // PASS 1: High-confidence IoU matching (IoU > 0.2)
        for (let t = 0; t < this.tracks.length; t++) {
            const track = this.tracks[t];
            let bestIdx = -1;
            let bestIoU = 0.2;

            for (let d = 0; d < detections.length; d++) {
                if (usedDetections.has(d)) continue;
                const iou = calculateIoU(track, detections[d]);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestIdx = d;
                }
            }

            if (bestIdx !== -1) {
                this.applyMatch(track, detections[bestIdx]);
                usedDetections.add(bestIdx);
                matchedTracks.add(t);
            }
        }

        // PASS 2: Distance-based fallback for unmatched tracks
        // This helps re-identify people who went off-screen and came back
        for (let t = 0; t < this.tracks.length; t++) {
            if (matchedTracks.has(t)) continue;
            const track = this.tracks[t];

            let bestIdx = -1;
            let bestDist = 150; // max pixel distance to consider a match

            for (let d = 0; d < detections.length; d++) {
                if (usedDetections.has(d)) continue;
                const det = detections[d];

                const detCx = (det.x1 + det.x2) / 2;
                const detCy = (det.y1 + det.y2) / 2;
                const trackCx = (track.x1 + track.x2) / 2;
                const trackCy = (track.y1 + track.y2) / 2;

                const dist = Math.sqrt((detCx - trackCx) ** 2 + (detCy - trackCy) ** 2);

                // Also check size similarity (reject if sizes differ wildly)
                const detArea = (det.x2 - det.x1) * (det.y2 - det.y1);
                const trackArea = (track.x2 - track.x1) * (track.y2 - track.y1);
                const sizeRatio = Math.min(detArea, trackArea) / Math.max(detArea, trackArea);

                if (dist < bestDist && sizeRatio > 0.4) {
                    bestDist = dist;
                    bestIdx = d;
                }
            }

            if (bestIdx !== -1) {
                this.applyMatch(track, detections[bestIdx]);
                usedDetections.add(bestIdx);
                matchedTracks.add(t);
            } else {
                track.missedFrames++;
            }
        }

        // Remove tracks that have been missing for too long
        this.tracks = this.tracks.filter(t => t.missedFrames < this.maxMissedFrames);

        // Add unmatched detections as new tracks
        for (let d = 0; d < detections.length; d++) {
            if (usedDetections.has(d)) continue;
            const det = detections[d];
            this.tracks.push({
                ...det,
                trackId: this.nextId++,
                missedFrames: 0,
                vx: 0,
                vy: 0,
                lastCx: (det.x1 + det.x2) / 2,
                lastCy: (det.y1 + det.y2) / 2,
            });
        }

        return this.getVisibleTracks();
    }

    /** Apply a detection match to an existing track, updating velocity */
    private applyMatch(track: TrackedObject, det: BoundingBox) {
        const newCx = (det.x1 + det.x2) / 2;
        const newCy = (det.y1 + det.y2) / 2;

        // Smooth velocity with exponential moving average (0.7 old + 0.3 new)
        if (track.missedFrames === 0) {
            track.vx = 0.7 * track.vx + 0.3 * (newCx - track.lastCx);
            track.vy = 0.7 * track.vy + 0.3 * (newCy - track.lastCy);
        }

        track.x1 = det.x1;
        track.y1 = det.y1;
        track.x2 = det.x2;
        track.y2 = det.y2;
        track.confidence = det.confidence;
        track.missedFrames = 0;
        track.lastCx = newCx;
        track.lastCy = newCy;
    }

    /** Only return tracks that were seen recently (not ghost predictions) */
    private getVisibleTracks(): TrackedObject[] {
        // Show tracks that were seen in the last 3 frames
        return this.tracks.filter(t => t.missedFrames <= 3);
    }
}
