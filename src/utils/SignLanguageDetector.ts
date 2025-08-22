import * as tf from '@tensorflow/tfjs';
import { MediaPipeDetector, HandLandmarks } from './MediaPipeDetector';
import { VideoTrainer } from './VideoTrainer';

export interface DetectionResult {
  gesture: string;
  confidence: number;
}

export class SignLanguageDetector {
  private model: tf.Sequential | null = null;
  private mediaPipe: MediaPipeDetector | null = null;
  private isInitialized = false;
  private labelMap: { [key: number]: string } = {
    0: 'hello',
    1: 'thankyou'
  };
  private sequenceBuffer: number[][] = [];
  private readonly SEQUENCE_LENGTH = 30;
  private readonly LANDMARK_COUNT = 42;
  private onDetection: ((result: DetectionResult | null) => void) | null = null;

  async initialize(): Promise<void> {
    try {
      console.log('🚀 Initializing Sign Language Detector...');
      
      // Initialize MediaPipe for real hand detection
      this.mediaPipe = new MediaPipeDetector();
      await this.mediaPipe.initialize();
      
      // Try to load pre-trained model, otherwise train new one
      await this.loadOrTrainModel();
      
      this.isInitialized = true;
      console.log('✅ Sign language detector initialized');
    } catch (error) {
      console.error('❌ Failed to initialize detector:', error);
      throw error;
    }
  }

  private async loadOrTrainModel(): Promise<void> {
    try {
      console.log('📥 Attempting to load pre-trained model...');
      this.model = await tf.loadLayersModel('/model/model.json') as tf.Sequential;
      console.log('✅ Pre-trained model loaded');
    } catch (error) {
      console.log('⚠️ Pre-trained model not found, training new model...');
      await this.trainNewModel();
    }
  }

  private async trainNewModel(): Promise<void> {
    const trainer = new VideoTrainer();
    
    // Load training data (simulating your video processing)
    const trainingData = await trainer.loadTrainingData();
    
    // Create and train model
    this.model = await trainer.createAndTrainModel(trainingData);
    
    console.log('✅ New model trained successfully');
  }

  startDetection(videoElement: HTMLVideoElement, onDetection: (result: DetectionResult | null) => void): void {
    if (!this.isInitialized || !this.mediaPipe) {
      console.error('❌ Detector not initialized');
      return;
    }

    this.onDetection = onDetection;
    this.sequenceBuffer = [];

    // Start MediaPipe camera with landmark detection
    this.mediaPipe.startCamera(videoElement, (landmarks: HandLandmarks | null) => {
      this.processLandmarks(landmarks);
    });

    console.log('🎯 Detection started with MediaPipe');
  }

  private processLandmarks(landmarks: HandLandmarks | null): void {
    if (!landmarks || !this.model) {
      if (this.onDetection) {
        this.onDetection(null);
      }
      return;
    }

    // Add landmarks to sequence buffer
    if (landmarks.landmarks.length > 0) {
      this.sequenceBuffer.push(landmarks.landmarks[0]);
      
      // Maintain buffer size
      if (this.sequenceBuffer.length > this.SEQUENCE_LENGTH) {
        this.sequenceBuffer.shift();
      }

      console.log(`📊 Sequence buffer: ${this.sequenceBuffer.length}/${this.SEQUENCE_LENGTH} frames`);

      // Make prediction when buffer is full
      if (this.sequenceBuffer.length === this.SEQUENCE_LENGTH) {
        this.makePrediction();
      }
    }
  }

  private async makePrediction(): Promise<void> {
    if (!this.model || this.sequenceBuffer.length < this.SEQUENCE_LENGTH) {
      return;
    }

    try {
      // Prepare input tensor
      const inputTensor = tf.tensor3d([this.sequenceBuffer]);
      const prediction = this.model.predict(inputTensor) as tf.Tensor;
      const predictionData = await prediction.data();
      
      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();

      // Get prediction results
      const maxIndex = predictionData.indexOf(Math.max(...Array.from(predictionData)));
      const confidence = predictionData[maxIndex];

      console.log(`🎯 Prediction: ${this.labelMap[maxIndex]} (${(confidence * 100).toFixed(1)}%)`);

      // Return result if confidence is high enough
      if (confidence > 0.6 && this.onDetection) {
        this.onDetection({
          gesture: this.labelMap[maxIndex],
          confidence: confidence
        });
      } else if (this.onDetection) {
        this.onDetection(null);
      }
    } catch (error) {
      console.error('❌ Error making prediction:', error);
    }
  }

  stopDetection(): void {
    if (this.mediaPipe) {
      this.mediaPipe.stopCamera();
    }
    this.sequenceBuffer = [];
    this.onDetection = null;
    console.log('⏹️ Detection stopped');
  }

  dispose(): void {
    this.stopDetection();
    
    if (this.mediaPipe) {
      this.mediaPipe.dispose();
      this.mediaPipe = null;
    }
    
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    
    this.isInitialized = false;
    console.log('🧹 SignLanguageDetector disposed');
  }
}