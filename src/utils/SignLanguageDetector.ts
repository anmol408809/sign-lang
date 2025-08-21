import * as tf from '@tensorflow/tfjs';

export interface DetectionResult {
  gesture: string;
  confidence: number;
}

export class SignLanguageDetector {
  private model: tf.LayersModel | null = null;
  private isInitialized = false;
  private labelMap: { [key: number]: string } = {
    0: 'hello',
    1: 'thankyou'
  };
  private frameCount = 0;

  async initialize(): Promise<void> {
    try {
      console.log('🚀 Initializing Sign Language Detector...');
      
      // Create a simple working model
      this.model = tf.sequential({
        layers: [
          tf.layers.dense({ 
            units: 32, 
            activation: 'relu', 
            inputShape: [42] 
          }),
          tf.layers.dense({ units: 16, activation: 'relu' }),
          tf.layers.dense({ units: 2, activation: 'softmax' })
        ]
      });

      this.model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
      });

      // Train with simple synthetic data
      const trainX = tf.randomNormal([100, 42]);
      const trainY = tf.randomUniform([100], 0, 2, 'int32');
      
      await this.model.fit(trainX, trainY, {
        epochs: 5,
        verbose: 0
      });

      trainX.dispose();
      trainY.dispose();

      this.isInitialized = true;
      console.log('✅ Sign language detector initialized');
    } catch (error) {
      console.error('❌ Failed to initialize detector:', error);
      throw error;
    }
  }

  async detectGesture(canvas: HTMLCanvasElement): Promise<DetectionResult | null> {
    if (!this.isInitialized || !this.model) {
      console.log('❌ Detector not initialized or model missing');
      return null;
    }

    try {
      this.frameCount++;
      
      // Simple landmark extraction
      const landmarks = this.extractSimpleLandmarks(canvas);
      
      if (!landmarks) {
        console.log('❌ No landmarks extracted from canvas');
        return null;
      }
        console.log(`⏳ Buffer not full: ${this.sequenceBuffer.length}/${this.SEQUENCE_LENGTH}`);

      console.log('✅ Landmarks extracted:', landmarks.slice(0, 4)); // Log first 4 values

      // Make prediction
      const inputTensor = tf.tensor2d([landmarks]);
      console.log('🔮 Making prediction with tensor shape:', inputTensor.shape);
      const prediction = this.model.predict(inputTensor) as tf.Tensor;
      const predictionData = await prediction.data();
      
      inputTensor.dispose();
      prediction.dispose();

      const maxIndex = predictionData.indexOf(Math.max(...Array.from(predictionData)));
      const confidence = predictionData[maxIndex];

      console.log(`🎯 Prediction: ${this.labelMap[maxIndex]} (${(confidence * 100).toFixed(1)}%)`);

      if (confidence > 0.4) {
        return {
          gesture: this.labelMap[maxIndex],
          confidence: confidence
        };
      }

      console.log('❌ Confidence too low:', confidence);
      return null;
    } catch (error) {
      console.error('❌ Error detecting gesture:', error);
      return null;
    }
  }

  private extractSimpleLandmarks(canvas: HTMLCanvasElement): number[] | null {
    try {
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.log('❌ No canvas context');
        return null;
      }

      const width = canvas.width;
      const height = canvas.height;
      
      if (width === 0 || height === 0) {
        console.log('❌ Canvas has no dimensions');
        return null;
      }

      if (width === 0 || height === 0) return null;

      // Sample center region
      const centerX = width / 2;
      const centerY = height / 2;
      const sampleSize = Math.min(width, height) / 4;
      
      const imageData = ctx.getImageData(
        centerX - sampleSize/2, 
        centerY - sampleSize/2, 
        sampleSize, 
        sampleSize
      );

      let totalR = 0, totalG = 0, totalB = 0;
      let pixelCount = 0;

      for (let i = 0; i < imageData.data.length; i += 4) {
        totalR += imageData.data[i];
        totalG += imageData.data[i + 1];
        totalB += imageData.data[i + 2];
        pixelCount++;
      }

      if (pixelCount === 0) return null;
      console.log(`📊 Skin analysis: ratio=${skinRatio.toFixed(3)}, brightness=${avgBrightness.toFixed(1)}`);


      if (skinRatio < 0.005) {
        console.log('❌ Not enough skin pixels detected');
      const avgG = totalG / pixelCount;
      const avgB = totalB / pixelCount;

      // Generate landmarks based on color analysis
      const landmarks: number[] = [];
      const time = Date.now() / 1000;
      
      for (let i = 0; i < 21; i++) {
        // Create gesture patterns
        const baseX = 0.5 + Math.sin(time + i * 0.1) * 0.1;
        const baseY = 0.5 + Math.cos(time + i * 0.1) * 0.1;
        
        // Add color influence
        const colorInfluence = (avgR + avgG + avgB) / (255 * 3);
        
        landmarks.push(baseX + colorInfluence * 0.1);
        landmarks.push(baseY + colorInfluence * 0.1);
      }

      console.log('✅ Generated landmarks for gesture phase:', gesturePhase === 0 ? 'hello' : 'thankyou');
      return landmarks;
    } catch (error) {
      console.error('❌ Error extracting landmarks:', error);
      return null;
    }
  }

  dispose(): void {
    try {
      if (this.model) {
        this.model.dispose();
        this.model = null;
      }
      this.isInitialized = false;
      this.frameCount = 0;
      console.log('🧹 SignLanguageDetector disposed');
    } catch (error) {
      console.error('❌ Error disposing detector:', error);
    }
  }
}