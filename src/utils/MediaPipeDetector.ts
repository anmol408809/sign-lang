import { Hands, Results } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

export interface HandLandmarks {
  landmarks: number[][];
  timestamp: number;
}

export class MediaPipeDetector {
  private hands: Hands | null = null;
  private camera: Camera | null = null;
  private isInitialized = false;
  private onResults: ((landmarks: HandLandmarks | null) => void) | null = null;

  async initialize(): Promise<void> {
    try {
      console.log('🤖 Initializing MediaPipe Hands...');
      
      // Initialize MediaPipe Hands
      this.hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });

      this.hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
      });

      this.hands.onResults((results: Results) => {
        this.processResults(results);
      });

      this.isInitialized = true;
      console.log('✅ MediaPipe Hands initialized');
    } catch (error) {
      console.error('❌ Failed to initialize MediaPipe:', error);
      throw error;
    }
  }

  private processResults(results: Results): void {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const handLandmarks = results.multiHandLandmarks[0];
      
      // Convert to our format (flatten x,y coordinates)
      const landmarks: number[][] = [];
      const currentFrame: number[] = [];
      
      for (const landmark of handLandmarks) {
        currentFrame.push(landmark.x);
        currentFrame.push(landmark.y);
      }
      
      landmarks.push(currentFrame);
      
      const handData: HandLandmarks = {
        landmarks: landmarks,
        timestamp: Date.now()
      };
      
      if (this.onResults) {
        this.onResults(handData);
      }
    } else {
      if (this.onResults) {
        this.onResults(null);
      }
    }
  }

  startCamera(videoElement: HTMLVideoElement, onResults: (landmarks: HandLandmarks | null) => void): void {
    if (!this.isInitialized || !this.hands) {
      console.error('❌ MediaPipe not initialized');
      return;
    }

    this.onResults = onResults;

    this.camera = new Camera(videoElement, {
      onFrame: async () => {
        if (this.hands) {
          await this.hands.send({ image: videoElement });
        }
      },
      width: 640,
      height: 480
    });

    this.camera.start();
    console.log('📹 MediaPipe camera started');
  }

  stopCamera(): void {
    if (this.camera) {
      this.camera.stop();
      this.camera = null;
    }
    this.onResults = null;
    console.log('⏹️ MediaPipe camera stopped');
  }

  dispose(): void {
    this.stopCamera();
    if (this.hands) {
      this.hands.close();
      this.hands = null;
    }
    this.isInitialized = false;
    console.log('🧹 MediaPipe detector disposed');
  }
}