import * as tf from '@tensorflow/tfjs';

export interface TrainingData {
  sequences: number[][][];
  labels: number[];
}

export class ModelTrainer {
  private model: tf.Sequential | null = null;
  private readonly SEQUENCE_LENGTH = 30;
  private readonly LANDMARK_COUNT = 42;
  private readonly NUM_CLASSES = 2;

  createModel(): tf.Sequential {
    console.log('🏗️ Creating LSTM model...');
    
    this.model = tf.sequential({
      layers: [
        // First LSTM layer with return sequences
        tf.layers.lstm({
          units: 128,
          returnSequences: true,
          inputShape: [this.SEQUENCE_LENGTH, this.LANDMARK_COUNT],
          name: 'lstm_1'
        }),
        tf.layers.dropout({ rate: 0.3, name: 'dropout_1' }),
        
        // Second LSTM layer
        tf.layers.lstm({
          units: 64,
          returnSequences: false,
          name: 'lstm_2'
        }),
        tf.layers.dropout({ rate: 0.3, name: 'dropout_2' }),
        
        // Dense layers
        tf.layers.dense({ 
          units: 64, 
          activation: 'relu',
          name: 'dense_1'
        }),
        tf.layers.dense({ 
          units: 32, 
          activation: 'relu',
          name: 'dense_2'
        }),
        tf.layers.dense({ 
          units: this.NUM_CLASSES, 
          activation: 'softmax',
          name: 'output'
        })
      ]
    });

    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });

    console.log('✅ LSTM model created');
    this.model.summary();
    
    return this.model;
  }

  async trainModel(trainingData: TrainingData, epochs: number = 50): Promise<tf.History> {
    if (!this.model) {
      throw new Error('Model not created. Call createModel() first.');
    }

    console.log('🚀 Starting model training...');
    console.log(`Training samples: ${trainingData.sequences.length}`);
    console.log(`Epochs: ${epochs}`);

    // Convert training data to tensors
    const xTrain = tf.tensor3d(trainingData.sequences);
    const yTrain = tf.tensor1d(trainingData.labels, 'int32');

    console.log('Input shape:', xTrain.shape);
    console.log('Output shape:', yTrain.shape);

    // Split data for validation
    const splitIndex = Math.floor(trainingData.sequences.length * 0.8);
    
    const xTrainSplit = xTrain.slice([0, 0, 0], [splitIndex, -1, -1]);
    const yTrainSplit = yTrain.slice([0], [splitIndex]);
    const xValSplit = xTrain.slice([splitIndex, 0, 0], [-1, -1, -1]);
    const yValSplit = yTrain.slice([splitIndex], [-1]);

    try {
      const history = await this.model.fit(xTrainSplit, yTrainSplit, {
        epochs: epochs,
        batchSize: 4,
        validationData: [xValSplit, yValSplit],
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs?.loss?.toFixed(4)} - Accuracy: ${logs?.acc?.toFixed(4)} - Val Loss: ${logs?.val_loss?.toFixed(4)} - Val Accuracy: ${logs?.val_acc?.toFixed(4)}`);
          }
        }
      });

      console.log('✅ Model training completed');
      return history;
    } finally {
      // Clean up tensors
      xTrain.dispose();
      yTrain.dispose();
      xTrainSplit.dispose();
      yTrainSplit.dispose();
      xValSplit.dispose();
      yValSplit.dispose();
    }
  }

  async saveModel(path: string = 'localstorage://sign-language-model'): Promise<void> {
    if (!this.model) {
      throw new Error('No model to save');
    }

    console.log('💾 Saving model...');
    await this.model.save(path);
    console.log('✅ Model saved successfully');
  }

  async loadModel(path: string = 'localstorage://sign-language-model'): Promise<tf.LayersModel> {
    console.log('📥 Loading model...');
    this.model = await tf.loadLayersModel(path) as tf.Sequential;
    console.log('✅ Model loaded successfully');
    return this.model;
  }

  getModel(): tf.Sequential | null {
    return this.model;
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}