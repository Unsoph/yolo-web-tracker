import { type BoundingBox, calculateIoU } from "./yoloUtils";

export interface TrackedObject extends BoundingBox {
    trackId: number;
    missedFrames: number;
}

/**
 * Clean IoU-only tracker optimised for MOVING CAMERAS (drones).
 * 
 * Key design choice: NO velocity prediction, NO distance fallback.
 * With a moving camera, all objects shift together when you pan.
 * Velocity prediction and distance matching cause ID swaps in that scenario.
 * Pure IoU matching is the most reliable because overlapping boxes
 * between frames almost always belong to the same object.
 * 
 * If an object leaves the frame, it gets a new ID when it comes back.
 * This is the correct trade-off: a new ID is better than a WRONG ID.
 */
export class SimpleTracker {
    private tracks: TrackedObject[] = [];
    private nextId: number = 1;
    private maxMissedFrames: number = 15; // ~0.5s at 30fps

    public update(detections: BoundingBox[]): TrackedObject[] {
        if (this.tracks.length === 0) {
            this.tracks = detections.map(det => ({
                ...det,
                trackId: this.nextId++,
                missedFrames: 0,
            }));
            return this.tracks.filter(t => t.missedFrames === 0);
        }

        // Build a cost matrix of IoU scores between all tracks and detections
        const iouMatrix: { trackIdx: number; detIdx: number; iou: number }[] = [];

        for (let t = 0; t < this.tracks.length; t++) {
            for (let d = 0; d < detections.length; d++) {
                const iou = calculateIoU(this.tracks[t], detections[d]);
                if (iou > 0.25) {  // minimum IoU to even consider a match
                    iouMatrix.push({ trackIdx: t, detIdx: d, iou });
                }
            }
        }

        // Greedy matching: highest IoU pairs first (prevents cross-assignments)
        iouMatrix.sort((a, b) => b.iou - a.iou);

        const matchedTracks = new Set<number>();
        const matchedDets = new Set<number>();

        for (const match of iouMatrix) {
            if (matchedTracks.has(match.trackIdx) || matchedDets.has(match.detIdx)) {
                continue; // already assigned
            }

            const track = this.tracks[match.trackIdx];
            const det = detections[match.detIdx];

            track.x1 = det.x1;
            track.y1 = det.y1;
            track.x2 = det.x2;
            track.y2 = det.y2;
            track.confidence = det.confidence;
            track.missedFrames = 0;

            matchedTracks.add(match.trackIdx);
            matchedDets.add(match.detIdx);
        }

        // Increment missed frames for unmatched tracks
        for (let t = 0; t < this.tracks.length; t++) {
            if (!matchedTracks.has(t)) {
                this.tracks[t].missedFrames++;
            }
        }

        // Remove stale tracks
        this.tracks = this.tracks.filter(t => t.missedFrames < this.maxMissedFrames);

        // Create new tracks for unmatched detections
        for (let d = 0; d < detections.length; d++) {
            if (!matchedDets.has(d)) {
                this.tracks.push({
                    ...detections[d],
                    trackId: this.nextId++,
                    missedFrames: 0,
                });
            }
        }

        // Only return tracks that are currently visible (seen this frame)
        return this.tracks.filter(t => t.missedFrames === 0);
    }
}
