import { type BoundingBox, calculateIoU } from "./yoloUtils";

export interface TrackedObject extends BoundingBox {
    trackId: number;
    missedFrames: number;
}

/**
 * IoU tracker with a special "sticky lock" mode for the selected target.
 * 
 * Regular tracks: matched by IoU only (prevents ID swaps on camera pan).
 * Locked target:  gets a distance+size fallback so it doesn't drop when
 *                 the person walks away and their box shrinks.
 */
export class SimpleTracker {
    private tracks: TrackedObject[] = [];
    private nextId: number = 1;
    private maxMissedFrames: number = 15;

    public update(detections: BoundingBox[], lockedTrackId: number | null = null): TrackedObject[] {
        if (this.tracks.length === 0) {
            this.tracks = detections.map(det => ({
                ...det,
                trackId: this.nextId++,
                missedFrames: 0,
            }));
            return this.tracks.filter(t => t.missedFrames === 0);
        }

        // Build IoU cost matrix
        const iouMatrix: { trackIdx: number; detIdx: number; iou: number }[] = [];

        for (let t = 0; t < this.tracks.length; t++) {
            for (let d = 0; d < detections.length; d++) {
                const iou = calculateIoU(this.tracks[t], detections[d]);
                if (iou > 0.15) {  // lowered from 0.25 for better continuity
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

        // SPECIAL: If the locked target wasn't matched by IoU, try distance+size fallback.
        // This ONLY applies to the locked track — not all tracks — so it won't cause
        // the ID-swap problem on camera pan.
        if (lockedTrackId !== null) {
            const lockedIdx = this.tracks.findIndex(t => t.trackId === lockedTrackId);
            if (lockedIdx !== -1 && !matchedTracks.has(lockedIdx)) {
                const track = this.tracks[lockedIdx];
                const trackCx = (track.x1 + track.x2) / 2;
                const trackCy = (track.y1 + track.y2) / 2;
                const trackArea = (track.x2 - track.x1) * (track.y2 - track.y1);

                let bestIdx = -1;
                let bestDist = 200; // generous pixel radius for locked target

                for (let d = 0; d < detections.length; d++) {
                    if (matchedDets.has(d)) continue;
                    const det = detections[d];
                    const detCx = (det.x1 + det.x2) / 2;
                    const detCy = (det.y1 + det.y2) / 2;
                    const detArea = (det.x2 - det.x1) * (det.y2 - det.y1);

                    const dist = Math.sqrt((detCx - trackCx) ** 2 + (detCy - trackCy) ** 2);
                    const sizeRatio = Math.min(detArea, trackArea) / Math.max(detArea, trackArea);

                    // Must be reasonably close AND similar size
                    if (dist < bestDist && sizeRatio > 0.3) {
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
