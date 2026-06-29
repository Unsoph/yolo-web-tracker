import { type BoundingBox, calculateIoU } from "./yoloUtils";

export interface TrackedObject extends BoundingBox {
    trackId: number;
    missedFrames: number;
}

/**
 * IoU tracker tuned for LOW FPS (~5-15) with sticky lock mode.
 * 
 * At 5 FPS each frame is 200ms apart — people move a LOT between frames.
 * All thresholds are set very permissively to account for this.
 */
export class SimpleTracker {
    private tracks: TrackedObject[] = [];
    private nextId: number = 1;
    private maxMissedFrames: number = 20;

    public update(detections: BoundingBox[], lockedTrackId: number | null = null): TrackedObject[] {
        if (this.tracks.length === 0) {
            this.tracks = detections.map(det => ({
                ...det,
                trackId: this.nextId++,
                missedFrames: 0,
            }));
            return this.tracks.filter(t => t.missedFrames === 0);
        }

        // Build IoU cost matrix with a very low threshold (5 FPS = large movement)
        const iouMatrix: { trackIdx: number; detIdx: number; iou: number }[] = [];

        for (let t = 0; t < this.tracks.length; t++) {
            for (let d = 0; d < detections.length; d++) {
                const iou = calculateIoU(this.tracks[t], detections[d]);
                if (iou > 0.05) {  // very low — even a tiny overlap counts at low FPS
                    iouMatrix.push({ trackIdx: t, detIdx: d, iou });
                }
            }
        }

        // Greedy matching: highest IoU first
        iouMatrix.sort((a, b) => b.iou - a.iou);

        const matchedTracks = new Set<number>();
        const matchedDets = new Set<number>();

        for (const match of iouMatrix) {
            if (matchedTracks.has(match.trackIdx) || matchedDets.has(match.detIdx)) continue;

            this.applyMatch(this.tracks[match.trackIdx], detections[match.detIdx]);
            matchedTracks.add(match.trackIdx);
            matchedDets.add(match.detIdx);
        }

        // STICKY LOCK: If the locked target wasn't matched by IoU,
        // use a generous distance + size fallback (ONLY for this one track).
        if (lockedTrackId !== null) {
            const lockedIdx = this.tracks.findIndex(t => t.trackId === lockedTrackId);
            if (lockedIdx !== -1 && !matchedTracks.has(lockedIdx)) {
                const track = this.tracks[lockedIdx];
                const trackCx = (track.x1 + track.x2) / 2;
                const trackCy = (track.y1 + track.y2) / 2;
                const trackArea = (track.x2 - track.x1) * (track.y2 - track.y1);

                let bestIdx = -1;
                let bestDist = 400; // very generous — at 5fps people move far

                for (let d = 0; d < detections.length; d++) {
                    if (matchedDets.has(d)) continue;
                    const det = detections[d];
                    const detCx = (det.x1 + det.x2) / 2;
                    const detCy = (det.y1 + det.y2) / 2;
                    const detArea = (det.x2 - det.x1) * (det.y2 - det.y1);

                    const dist = Math.sqrt((detCx - trackCx) ** 2 + (detCy - trackCy) ** 2);
                    const sizeRatio = Math.min(detArea, trackArea) / Math.max(detArea, trackArea);

                    if (dist < bestDist && sizeRatio > 0.1) {
                        bestDist = dist;
                        bestIdx = d;
                    }
                }

                if (bestIdx !== -1) {
                    this.applyMatch(track, detections[bestIdx]);
                    matchedTracks.add(lockedIdx);
                    matchedDets.add(bestIdx);
                }
            }
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

        return this.tracks.filter(t => t.missedFrames === 0);
    }

    private applyMatch(track: TrackedObject, det: BoundingBox) {
        track.x1 = det.x1;
        track.y1 = det.y1;
        track.x2 = det.x2;
        track.y2 = det.y2;
        track.confidence = det.confidence;
        track.missedFrames = 0;
    }
}
