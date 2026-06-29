import { type BoundingBox, calculateIoU } from "./yoloUtils";

export interface TrackedObject extends BoundingBox {
    trackId: number;
    missedFrames: number;
}

/**
 * A lightweight custom IoU Tracker.
 * Instead of complex Kalman filters, this tracker assigns consistent IDs across frames
 * by matching new detections to existing tracks based on bounding box overlap (IoU).
 * It runs extremely fast in Javascript.
 */
export class SimpleTracker {
    private tracks: TrackedObject[] = [];
    private nextId: number = 1;
    private maxMissedFrames: number = 5; // Track is deleted if lost for 5 frames

    public update(detections: BoundingBox[]): TrackedObject[] {
        // If this is the first frame or we lost all tracks
        if (this.tracks.length === 0) {
            this.tracks = detections.map(det => ({
                ...det,
                trackId: this.nextId++,
                missedFrames: 0
            }));
            return this.tracks;
        }

        const unmatchedDetections = [...detections];
        
        // Try to match existing tracks with new detections
        for (const track of this.tracks) {
            let bestMatchIdx = -1;
            let highestIoU = 0.2; // Minimum IoU required to consider it a match

            for (let i = 0; i < unmatchedDetections.length; i++) {
                const det = unmatchedDetections[i];
                
                // Only match if they are the same class (e.g. Person to Person)
                if (det.classId !== track.classId) continue;

                const iou = calculateIoU(track, det);
                if (iou > highestIoU) {
                    highestIoU = iou;
                    bestMatchIdx = i;
                }
            }

            if (bestMatchIdx !== -1) {
                // We found a match! Update track with new box coordinates
                const match = unmatchedDetections[bestMatchIdx];
                track.x1 = match.x1;
                track.y1 = match.y1;
                track.x2 = match.x2;
                track.y2 = match.y2;
                track.confidence = match.confidence;
                track.missedFrames = 0;
                
                // Remove from unmatched list so it can't be matched again
                unmatchedDetections.splice(bestMatchIdx, 1);
            } else {
                // The track wasn't found in this frame
                track.missedFrames++;
            }
        }

        // Remove tracks that have been missing for too long
        this.tracks = this.tracks.filter(t => t.missedFrames < this.maxMissedFrames);

        // Add any remaining unmatched detections as brand new tracks
        for (const det of unmatchedDetections) {
            this.tracks.push({
                ...det,
                trackId: this.nextId++,
                missedFrames: 0
            });
        }

        return this.tracks;
    }
}
