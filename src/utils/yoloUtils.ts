export interface BoundingBox {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
    classId: number;
    colorPatch?: number[];
}

/**
 * Parses the raw Float32Array output from YOLOv8 ONNX model.
 * OPTIMISED: Only checks class 0 (person) instead of looping all 80 classes.
 * Uses 320x320 input for speed.
 */
export function processYoloOutput(
    outputData: Float32Array, 
    confidenceThreshold: number = 0.45,
    iouThreshold: number = 0.3,
    numAnchors: number = 6300  // 320x320 produces 6300 anchors
): BoundingBox[] {
    const boxes: BoundingBox[] = [];

    // OPTIMISATION: We only care about person (class 0).
    // Instead of looping through all 80 classes for each of 6300 anchors,
    // we directly read index for class 0 = offset 4.
    const personOffset = 4 * numAnchors;

    for (let i = 0; i < numAnchors; i++) {
        const conf = outputData[personOffset + i];

        if (conf > confidenceThreshold) {
            const cx = outputData[i];
            const cy = outputData[numAnchors + i];
            const w  = outputData[2 * numAnchors + i];
            const h  = outputData[3 * numAnchors + i];

            boxes.push({
                x1: cx - w / 2,
                y1: cy - h / 2,
                x2: cx + w / 2,
                y2: cy + h / 2,
                confidence: conf,
                classId: 0
            });
        }
    }

    return applyNMS(boxes, iouThreshold);
}

/**
 * Non-Maximum Suppression to remove overlapping duplicate boxes.
 */
function applyNMS(boxes: BoundingBox[], iouThreshold: number): BoundingBox[] {
    boxes.sort((a, b) => b.confidence - a.confidence);

    const result: BoundingBox[] = [];
    
    for (const box of boxes) {
        let suppress = false;
        for (const resBox of result) {
            if (calculateIoU(box, resBox) > iouThreshold) {
                suppress = true;
                break;
            }
        }
        if (!suppress) {
            result.push(box);
        }
    }
    return result;
}

export function calculateIoU(a: BoundingBox, b: BoundingBox): number {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    
    const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
    const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    const unionArea = aArea + bArea - intersectionArea;
    if (unionArea === 0) return 0;
    
    return intersectionArea / unionArea;
}
