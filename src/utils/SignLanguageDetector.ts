@@ .. @@
 import * as tf from '@tensorflow/tfjs';
+import { Hands, Results } from '@mediapipe/hands';
+import { Camera } from '@mediapipe/camera_utils';

 export interface DetectionResult {
   gesture: string;
   confidence: number;
 }

 export class SignLanguageDetector {
   private model: tf.LayersModel | null = null;
+  private hands: Hands | null = null;
   private isInitialized = false;
   private labelMap: { [key: number]: string } = {
     0: 'hello',
     1: 'thankyou'
   };
   private sequenceBuffer: number[][] = [];
   private readonly SEQUENCE_LENGTH = 30;
   private readonly LANDMARK_COUNT = 42;
   private frameCount = 0;

   async initialize(): Promise<void> {
     try {
       console.log('🚀 Initializing Sign Language Detector...');
       
-      // Create a working model that actually makes predictions
-      await this.createWorkingModel();
+      // Initialize MediaPipe Hands
+      await this.initializeMediaPipe();
+      
+      // Try to load trained model, fallback to creating a new one
+      await this.loadOrCreateModel();
       
       this.isInitialized = true;
       console.log('✅ Sign language detector initialized successfully');
     } catch (error) {
       console.error('❌ Failed to initialize sign language detector:', error);
       throw error;
     }
   }

-  private async createWorkingModel(): Promise<void> {
+  private async initializeMediaPipe(): Promise<void> {
+    try {
+      console.log('🤖 Initializing MediaPipe Hands...');
+      
+      this.hands = new Hands({
+        locateFile: (file) => {
+          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
+        }
+      });
+
+      this.hands.setOptions({
+        maxNumHands: 1,
+        modelComplexity: 1,
+        minDetectionConfidence: 0.7,
+        minTrackingConfidence: 0.5
+      });
+
+      console.log('✅ MediaPipe Hands initialized');
+    } catch (error) {
+      console.error('❌ Error initializing MediaPipe:', error);
+      throw error;
+    }
+  }

+  private async loadOrCreateModel(): Promise<void> {
     try {
-      console.log('🔧 Creating working model...');
+      console.log('📥 Attempting to load trained model...');
       
-      // Create a simple but functional model
-      this.model = tf.sequential({
-        layers: [
-          tf.layers.dense({ 
-            units: 64, 
-            activation: 'relu', 
-            inputShape: [this.LANDMARK_COUNT] 
-          }),
-          tf.layers.dropout({ rate: 0.2 }),
-          tf.layers.dense({ units: 32, activation: 'relu' }),
-          tf.layers.dense({ units: 2, activation: 'softmax' })
-        ]
-      });
+      try {
+        // Try to load from localStorage first
+        this.model = await tf.loadLayersModel('localstorage://sign-language-model');
+        console.log('✅ Loaded trained model from localStorage');
+      } catch (localError) {
+        console.log('📦 No local model found, trying to load from public folder...');
+        
+        try {
+          // Try to load from public folder
+          this.model = await tf.loadLayersModel('/model/model.json');
+          console.log('✅ Loaded model from public folder');
+        } catch (publicError) {
+          console.log('🔧 No pre-trained model found, creating new LSTM model...');
+          await this.createLSTMModel();
+        }
+      }
+    } catch (error) {
+      console.error('❌ Error loading/creating model:', error);
+      throw error;
+    }
+  }

+  private async createLSTMModel(): Promise<void> {
+    console.log('🏗️ Creating LSTM model...');
+    
+    this.model = tf.sequential({
+      layers: [
+        // LSTM layers for sequence processing
+        tf.layers.lstm({
+          units: 128,
+          returnSequences: true,
+          inputShape: [this.SEQUENCE_LENGTH, this.LANDMARK_COUNT],
+          name: 'lstm_1'
+        }),
+        tf.layers.dropout({ rate: 0.3 }),
+        
+        tf.layers.lstm({
+          units: 64,
+          returnSequences: false,
+          name: 'lstm_2'
+        }),
+        tf.layers.dropout({ rate: 0.3 }),
+        
+        // Dense layers
+        tf.layers.dense({ units: 64, activation: 'relu' }),
+        tf.layers.dense({ units: 32, activation: 'relu' }),
+        tf.layers.dense({ units: 2, activation: 'softmax' })
+      ]
+    });

     this.model.compile({
       optimizer: 'adam',
-      loss: 'sparseCategoricalCrossentropy',
+      loss: 'categoricalCrossentropy',
       metrics: ['accuracy']
     });

-    // Warm up the model
-    const warmupInput = tf.randomNormal([1, this.LANDMARK_COUNT]);
+    // Warm up the model with correct input shape
+    const warmupInput = tf.randomNormal([1, this.SEQUENCE_LENGTH, this.LANDMARK_COUNT]);
     const warmupOutput = this.model.predict(warmupInput) as tf.Tensor;
     warmupInput.dispose();
     warmupOutput.dispose();

-    console.log('✅ Working model created and warmed up');
-  } catch (error) {
-    console.error('❌ Error creating model:', error);
-    throw error;
-  }
+    console.log('✅ LSTM model created and warmed up');
   }

   async detectGesture(canvas: HTMLCanvasElement): Promise<DetectionResult | null> {
-    if (!this.isInitialized || !this.model) {
+    if (!this.isInitialized || !this.model || !this.hands) {
       return null;
     }

     try {
       this.frameCount++;
       
-      // Extract landmarks from canvas
-      const landmarks = this.extractLandmarksFromCanvas(canvas);
+      // Extract landmarks using MediaPipe
+      const landmarks = await this.extractLandmarksWithMediaPipe(canvas);
       
       if (!landmarks) {
         return null;
       }

-      // Make prediction with current landmarks
-      const inputTensor = tf.tensor2d([landmarks]);
-      const prediction = this.model.predict(inputTensor) as tf.Tensor;
-      const predictionData = await prediction.data();
+      // Add landmarks to sequence buffer
+      this.sequenceBuffer.push(landmarks);
+      
+      // Keep only the last SEQUENCE_LENGTH frames
+      if (this.sequenceBuffer.length > this.SEQUENCE_LENGTH) {
+        this.sequenceBuffer.shift();
+      }
+      
+      // Only make predictions when we have enough frames
+      if (this.sequenceBuffer.length < this.SEQUENCE_LENGTH) {
+        return null;
+      }
+
+      // Prepare input tensor for LSTM model
+      const inputTensor = tf.tensor3d([this.sequenceBuffer]);
+      const prediction = this.model.predict(inputTensor) as tf.Tensor;
+      const predictionData = await prediction.data();
       
       // Clean up tensors immediately
       inputTensor.dispose();
       prediction.dispose();

       // Get the class with highest probability
       const maxIndex = predictionData.indexOf(Math.max(...Array.from(predictionData)));
       const confidence = predictionData[maxIndex];

       // Log prediction for debugging
       if (this.frameCount % 30 === 0) { // Log every 30 frames
         console.log(`🎯 Frame ${this.frameCount}: ${this.labelMap[maxIndex]} (${(confidence * 100).toFixed(1)}%)`);
       }

-      // Return prediction if confidence is reasonable
-      if (confidence > 0.4) {
+      // Return prediction if confidence is high enough
+      if (confidence > 0.6) {
         return {
           gesture: this.labelMap[maxIndex],
           confidence: confidence
         };
       }

       return null;
     } catch (error) {
       console.error('❌ Error detecting gesture:', error);
       return null;
     }
   }

-  private extractLandmarksFromCanvas(canvas: HTMLCanvasElement): number[] | null {
+  private async extractLandmarksWithMediaPipe(canvas: HTMLCanvasElement): Promise<number[] | null> {
+    if (!this.hands) return null;
+    
     try {
-      const ctx = canvas.getContext('2d');
-      if (!ctx) return null;
-
-      // Analyze canvas for hand-like features
-      const width = canvas.width;
-      const height = canvas.height;
-      
-      // Sample multiple regions of the canvas
-      const regions = [
-        { x: width * 0.2, y: height * 0.2, w: width * 0.3, h: height * 0.3 },
-        { x: width * 0.5, y: height * 0.3, w: width * 0.3, h: height * 0.4 },
-        { x: width * 0.3, y: height * 0.4, w: width * 0.4, h: height * 0.3 }
-      ];
-
-      let totalSkinPixels = 0;
-      let totalPixels = 0;
-      let avgBrightness = 0;
-      let avgHue = 0;
-
-      for (const region of regions) {
-        const imageData = ctx.getImageData(
-          Math.max(0, region.x), 
-          Math.max(0, region.y), 
-          Math.min(region.w, width - region.x), 
-          Math.min(region.h, height - region.y)
-        );
-
-        let regionSkinPixels = 0;
-        let regionBrightness = 0;
-        let regionHue = 0;
-
-        for (let i = 0; i < imageData.data.length; i += 4) {
-          const r = imageData.data[i];
-          const g = imageData.data[i + 1];
-          const b = imageData.data[i + 2];
-          
-          // Enhanced skin detection
-          const isSkin = this.isSkinColor(r, g, b);
-          if (isSkin) {
-            regionSkinPixels++;
-          }
-          
-          regionBrightness += (r + g + b) / 3;
-          regionHue += Math.atan2(Math.sqrt(3) * (g - b), 2 * r - g - b);
-          totalPixels++;
-        }
-
-        totalSkinPixels += regionSkinPixels;
-        avgBrightness += regionBrightness;
-        avgHue += regionHue;
-      }
-
-      if (totalPixels === 0) return null;
-
-      avgBrightness /= totalPixels;
-      avgHue /= totalPixels;
-      const skinRatio = totalSkinPixels / totalPixels;
-
-      // Only proceed if we detect enough skin-like pixels
-      if (skinRatio < 0.01) {
-        return null;
-      }
-
-      // Generate realistic landmarks based on detected features
-      const landmarks: number[] = [];
-      const time = Date.now() / 1000;
-      
-      // Create gesture patterns that change over time
-      const gesturePhase = Math.floor(time / 4) % 2; // Switch every 4 seconds
-      const gestureProgress = (time % 4) / 4; // Progress within current gesture
-      
-      for (let i = 0; i < 21; i++) {
-        let x, y;
-        
-        if (gesturePhase === 0) {
-          // "Hello" gesture pattern - hand waving
-          x = 0.4 + 0.2 * Math.sin(time * 2 + i * 0.1) + skinRatio * 0.1;
-          y = 0.3 + 0.3 * Math.cos(time * 1.5 + i * 0.15) + (avgBrightness / 255) * 0.1;
-        } else {
-          // "Thank you" gesture pattern - more stable
-          x = 0.5 + 0.1 * Math.sin(time * 0.5 + i * 0.2) + skinRatio * 0.05;
-          y = 0.4 + 0.2 * Math.cos(time * 0.3 + i * 0.25) + (avgHue + 1) * 0.05;
-        }
-        
-        // Add some noise and clamp values
-        x += (Math.random() - 0.5) * 0.05;
-        y += (Math.random() - 0.5) * 0.05;
-        
-        landmarks.push(Math.max(0, Math.min(1, x)));
-        landmarks.push(Math.max(0, Math.min(1, y)));
-      }
-
-      return landmarks;
+      return new Promise((resolve) => {
+        this.hands!.onResults((results: Results) => {
+          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
+            const handLandmarks = results.multiHandLandmarks[0];
+            const landmarks: number[] = [];
+            
+            // Extract x, y coordinates for all 21 landmarks
+            for (const landmark of handLandmarks) {
+              landmarks.push(landmark.x);
+              landmarks.push(landmark.y);
+            }
+            
+            resolve(landmarks);
+          } else {
+            resolve(null);
+          }
+        });
+        
+        // Send canvas to MediaPipe
+        this.hands!.send({ image: canvas });
+      });
     } catch (error) {
       console.error('❌ Error extracting landmarks:', error);
       return null;
     }
   }

-  private isSkinColor(r: number, g: number, b: number): boolean {
-    // Multiple skin color detection methods
-    const method1 = r > 95 && g > 40 && b > 20 && 
-                   r > g && r > b && 
-                   Math.abs(r - g) > 15;
-    
-    const method2 = r > 220 && g > 210 && b > 170 &&
-                   Math.abs(r - g) <= 15 &&
-                   r > b && g > b;
-    
-    const method3 = r > 95 && g > 40 && b > 20 &&
-                   r - g > 15 && r - b > 15 &&
-                   r < 255 && g < 255 && b < 255;
-    
-    return method1 || method2 || method3;
-  }
-
   dispose(): void {
     try {
       if (this.model) {
         this.model.dispose();
         this.model = null;
       }
       
+      if (this.hands) {
+        this.hands.close();
+        this.hands = null;
+      }
+      
       this.sequenceBuffer = [];
       this.isInitialized = false;
       this.frameCount = 0;
       
       console.log('🧹 SignLanguageDetector disposed');
     } catch (error) {
       console.error('❌ Error disposing detector:', error);
     }
   }
 }