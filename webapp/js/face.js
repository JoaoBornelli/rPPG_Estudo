/**
 * face.js — MediaPipe Face Landmarker wrapper
 */

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

let landmarker = null;

export async function initFaceLandmarker(onProgress) {
  onProgress?.("Carregando módulos de visão...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  onProgress?.("Carregando modelo facial...");
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./assets/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  onProgress?.("Pronto");
  return landmarker;
}

/**
 * Detect face landmarks for a video frame.
 * @param {HTMLVideoElement} video
 * @param {number} timestampMs
 * @returns {Array|null} array of 478 landmarks [{x,y,z}] or null
 */
export function detectFace(video, timestampMs) {
  if (!landmarker) return null;
  const result = landmarker.detectForVideo(video, timestampMs);
  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    return result.faceLandmarks[0];
  }
  return null;
}
