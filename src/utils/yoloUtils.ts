export interface BoundingBox {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
    classId: number;
}

/**
 * Parses the raw Float32Array output from YOLOv8 ONNX model.
 * The ONNX YOLOv8 model outputs a tensor of shape [1, 84, 8400].
 */
export function processYoloOutput(
    outputData: Float32Array, 
    confidenceThreshold: number = 0.5,
    iouThreshold: number = 0.3
): BoundingBox[] {
    const boxes: BoundingBox[] = [];
    const numClasses = 80;
    const numAnchors = 8400; // number of predictions

    // The tensor layout is [1, 84, 8400] flattened to a 1D array.
    // For a specific anchor i and element e, index = e * numAnchors + i
    for (let i = 0; i < numAnchors; i++) {
        // Find the class with the highest probability
        let maxClassConf = 0;
        let classId = -1;
        
        for (let c = 0; c < numClasses; c++) {
            const conf = outputData[(4 + c) * numAnchors + i];
            if (conf > maxClassConf) {
                maxClassConf = conf;
                classId = c;
            }
        }

        if (maxClassConf > confidenceThreshold && classId === 0) {
            const cx = outputData[0 * numAnchors + i];
            const cy = outputData[1 * numAnchors + i];
            const w = outputData[2 * numAnchors + i];
            const h = outputData[3 * numAnchors + i];

            const x1 = cx - w / 2;
            const y1 = cy - h / 2;
            const x2 = cx + w / 2;
            const y2 = cy + h / 2;

            boxes.push({ x1, y1, x2, y2, confidence: maxClassConf, classId });
        }
    }

    return applyNMS(boxes, iouThreshold);
}

/**
 * Non-Maximum Suppression to remove overlapping duplicate boxes.
 */
function applyNMS(boxes: BoundingBox[], iouThreshold: number): BoundingBox[] {
    // Sort boxes by confidence, descending
    boxes.sort((a, b) => b.confidence - a.confidence);

    const result: BoundingBox[] = [];
    
    for (const box of boxes) {
        let suppress = false;
        for (const resBox of result) {
            if (box.classId !== resBox.classId) continue;
            
            const iou = calculateIoU(box, resBox);
            if (iou > iouThreshold) {
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
