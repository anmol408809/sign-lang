import React, { useState, useCallback } from 'react';
import { Play, Download, Upload, Brain, BarChart3 } from 'lucide-react';
import { ModelTrainer } from '../utils/ModelTrainer';
import { DataProcessor } from '../utils/DataProcessor';

const TrainingInterface: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [modelAccuracy, setModelAccuracy] = useState<number | null>(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(50);

  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setTrainingLogs(prev => [...prev.slice(-9), `[${timestamp}] ${message}`]);
  }, []);

  const startTraining = useCallback(async () => {
    if (isTraining) return;

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingLogs([]);
    setModelAccuracy(null);
    setCurrentEpoch(0);

    try {
      addLog('🚀 Starting model training...');
      
      // Initialize components
      const dataProcessor = new DataProcessor();
      const modelTrainer = new ModelTrainer();

      // Process video data
      addLog('📹 Processing video data...');
      const rawSequences = await dataProcessor.processVideoData();
      
      // Augment data
      addLog('📈 Augmenting training data...');
      const augmentedSequences = dataProcessor.augmentData(rawSequences);
      
      // Encode labels
      addLog('🏷️ Encoding labels...');
      const trainingData = dataProcessor.encodeLabels(augmentedSequences);
      
      addLog(`✅ Prepared ${trainingData.sequences.length} training samples`);

      // Create model
      addLog('🏗️ Creating LSTM model...');
      modelTrainer.createModel();

      // Train model with progress tracking
      addLog(`🎯 Training for ${totalEpochs} epochs...`);
      
      const history = await modelTrainer.trainModel(trainingData, totalEpochs);
      
      // Get final accuracy
      const finalAccuracy = history.history.acc?.[history.history.acc.length - 1] as number;
      setModelAccuracy(finalAccuracy);

      // Save model
      addLog('💾 Saving trained model...');
      await modelTrainer.saveModel();

      addLog('✅ Training completed successfully!');
      addLog(`🎯 Final accuracy: ${(finalAccuracy * 100).toFixed(2)}%`);

    } catch (error) {
      console.error('Training error:', error);
      addLog(`❌ Training failed: ${error}`);
    } finally {
      setIsTraining(false);
      setTrainingProgress(100);
    }
  }, [isTraining, totalEpochs, addLog]);

  const downloadModel = useCallback(async () => {
    try {
      addLog('📥 Downloading model...');
      // This would trigger a download of the trained model
      addLog('✅ Model download started');
    } catch (error) {
      addLog(`❌ Download failed: ${error}`);
    }
  }, [addLog]);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="glass-effect rounded-2xl p-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h2>
            <p className="text-gray-600">Train LSTM model on sign language video data</p>
          </div>
          <Brain className="w-12 h-12 text-primary-600" />
        </div>

        {/* Training Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-primary-50 rounded-xl p-4">
            <div className="text-2xl font-bold text-primary-600">10</div>
            <div className="text-sm text-gray-600">Training Videos</div>
            <div className="text-xs text-gray-500 mt-1">5 hello + 5 thankyou</div>
          </div>
          <div className="bg-accent-50 rounded-xl p-4">
            <div className="text-2xl font-bold text-accent-600">{totalEpochs}</div>
            <div className="text-sm text-gray-600">Training Epochs</div>
            <input 
              type="range" 
              min="10" 
              max="100" 
              value={totalEpochs}
              onChange={(e) => setTotalEpochs(parseInt(e.target.value))}
              disabled={isTraining}
              className="w-full mt-2"
            />
          </div>
          <div className="bg-purple-50 rounded-xl p-4">
            <div className="text-2xl font-bold text-purple-600">LSTM</div>
            <div className="text-sm text-gray-600">Model Architecture</div>
            <div className="text-xs text-gray-500 mt-1">128→64 units + Dense</div>
          </div>
        </div>

        {/* Training Progress */}
        {isTraining && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Training Progress</span>
              <span className="text-sm text-gray-500">Epoch {currentEpoch}/{totalEpochs}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-primary-500 to-accent-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Model Accuracy */}
        {modelAccuracy !== null && (
          <div className="mb-8 p-4 bg-accent-50 rounded-xl border border-accent-200">
            <div className="flex items-center space-x-2 mb-2">
              <BarChart3 className="w-5 h-5 text-accent-600" />
              <span className="font-semibold text-accent-800">Training Results</span>
            </div>
            <div className="text-2xl font-bold text-accent-600">
              {(modelAccuracy * 100).toFixed(2)}% Accuracy
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 mb-8">
          <button
            onClick={startTraining}
            disabled={isTraining}
            className={`flex items-center justify-center space-x-2 py-3 px-6 rounded-xl font-semibold transition-all duration-200 ${
              isTraining 
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                : 'bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 text-white shadow-lg hover:shadow-xl'
            }`}
          >
            {isTraining ? (
              <>
                <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                <span>Training...</span>
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                <span>Start Training</span>
              </>
            )}
          </button>

          <button
            onClick={downloadModel}
            disabled={!modelAccuracy}
            className="flex items-center justify-center space-x-2 py-3 px-6 bg-accent-100 hover:bg-accent-200 disabled:bg-gray-50 disabled:text-gray-400 text-accent-700 rounded-xl transition-colors font-semibold"
          >
            <Download className="w-5 h-5" />
            <span>Download Model</span>
          </button>
        </div>

        {/* Training Logs */}
        <div className="bg-gray-900 rounded-xl p-4">
          <div className="flex items-center space-x-2 mb-3">
            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-green-400 font-medium">Training Console</span>
          </div>
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {trainingLogs.length > 0 ? (
              trainingLogs.map((log, index) => (
                <div key={index} className="text-sm text-gray-300 font-mono">
                  {log}
                </div>
              ))
            ) : (
              <div className="text-gray-500 text-sm">Ready to start training...</div>
            )}
          </div>
        </div>

        {/* Architecture Info */}
        <div className="mt-8 p-6 bg-blue-50 rounded-xl border border-blue-200">
          <h3 className="font-semibold text-blue-900 mb-3">Model Architecture</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <div className="font-medium text-blue-800 mb-2">Input Layer</div>
              <ul className="text-blue-700 space-y-1">
                <li>• Sequence Length: 30 frames</li>
                <li>• Features: 42 landmarks (21 × 2)</li>
                <li>• Shape: (30, 42)</li>
              </ul>
            </div>
            <div>
              <div className="font-medium text-blue-800 mb-2">Hidden Layers</div>
              <ul className="text-blue-700 space-y-1">
                <li>• LSTM: 128 units → 64 units</li>
                <li>• Dropout: 30% each layer</li>
                <li>• Dense: 64 → 32 → 2 units</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingInterface;