import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { SignLanguageDetector, DetectionResult } from '../utils/SignLanguageDetector';

interface Prediction {
  gesture: string;
  confidence: number;
  timestamp: number;
}

interface SignLanguageContextType {
  isDetecting: boolean;
  currentPrediction: string | null;
  confidence: number;
  predictionHistory: Prediction[];
  modelLoaded: boolean;
  modelLoadingProgress: number;
  startDetection: () => void;
  stopDetection: () => void;
  startMediaPipeDetection: (videoElement: HTMLVideoElement) => void;
  clearHistory: () => void;
}

const SignLanguageContext = createContext<SignLanguageContextType | undefined>(undefined);

export const useSignLanguage = () => {
  const context = useContext(SignLanguageContext);
  if (!context) {
    throw new Error('useSignLanguage must be used within a SignLanguageProvider');
  }
  return context;
};

export const SignLanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [predictionHistory, setPredictionHistory] = useState<Prediction[]>([]);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelLoadingProgress, setModelLoadingProgress] = useState(0);
  
  const detectorRef = useRef<SignLanguageDetector | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const initializingRef = useRef<boolean>(false);

  // Initialize detector once
  useEffect(() => {
    let isMounted = true;
    
    const initializeDetector = async () => {
      if (initializingRef.current) return;
      initializingRef.current = true;
      
      try {
        console.log('🚀 Starting detector initialization...');
        
        // Simulate loading progress
        let progress = 0;
        const progressInterval = setInterval(() => {
          if (!isMounted) {
            clearInterval(progressInterval);
            return;
          }
          
          progress += Math.random() * 15;
          if (progress >= 90) {
            progress = 90;
            clearInterval(progressInterval);
          }
          setModelLoadingProgress(progress);
        }, 200);

        const newDetector = new SignLanguageDetector();
        await newDetector.initialize();
        
        if (isMounted) {
          detectorRef.current = newDetector;
          setModelLoaded(true);
          setModelLoadingProgress(100);
          console.log('✅ Detector with MediaPipe initialized successfully');
        } else {
          newDetector.dispose();
        }
        
        clearInterval(progressInterval);
      } catch (error) {
        console.error('❌ Failed to initialize detector:', error);
        if (isMounted) {
          setModelLoadingProgress(0);
        }
      } finally {
        initializingRef.current = false;
      }
    };

    initializeDetector();

    return () => {
      isMounted = false;
      if (detectorRef.current) {
        detectorRef.current.dispose();
        detectorRef.current = null;
      }
    };
  }, []);

  const startDetection = useCallback(() => {
    if (modelLoaded && detectorRef.current) {
      console.log('🎯 Starting MediaPipe detection...');
      setIsDetecting(true);
    } else {
      console.warn('⚠️ Cannot start detection: model not loaded');
    }
  }, [modelLoaded]);

  const stopDetection = useCallback(() => {
    console.log('⏹️ Stopping MediaPipe detection...');
    setIsDetecting(false);
    setCurrentPrediction(null);
    setConfidence(0);
    
    if (detectorRef.current) {
      detectorRef.current.stopDetection();
    }
  }, []);

  const startMediaPipeDetection = useCallback((videoElement: HTMLVideoElement) => {
    if (!detectorRef.current || !isDetecting || !modelLoaded) {
      console.log('❌ Cannot process frame:', { 
        detector: !!detectorRef.current, 
        detecting: isDetecting, 
    videoRef.current = videoElement;
    
    detectorRef.current.startDetection(videoElement, (result: DetectionResult | null) => {
      if (result && result.confidence > 0.6) {
        console.log(`🎯 DETECTED: ${result.gesture} (${(result.confidence * 100).toFixed(1)}%)`);
        
        setCurrentPrediction(result.gesture);
        setConfidence(result.confidence);
        
        // Add to history
        const newPrediction: Prediction = {
          gesture: result.gesture,
          confidence: result.confidence,
          timestamp: Date.now()
        };
        
        setPredictionHistory(prev => {
          // Avoid duplicates within 2 seconds
          const lastPrediction = prev[prev.length - 1];
          if (lastPrediction && 
              lastPrediction.gesture === newPrediction.gesture && 
              newPrediction.timestamp - lastPrediction.timestamp < 2000) {
            return prev;
          }
          
          // Keep only last 50 predictions
          const updated = [...prev, newPrediction];
          return updated.slice(-50);
        });
      } else {
        // Gradual confidence reduction
        setConfidence(prev => Math.max(0, prev * 0.9));
        
        if (confidence < 0.4) {
          setCurrentPrediction(null);
        }
      } else {
        if (result) {
          console.log(`❌ Low confidence: ${result.gesture} (${(result.confidence * 100).toFixed(1)}%)`);
      }
    });
  }, [isDetecting, modelLoaded, confidence]);

  const clearHistory = useCallback(() => {
    setPredictionHistory([]);
    console.log('🗑️ Prediction history cleared');
  }, []);

  const value: SignLanguageContextType = {
    isDetecting,
    currentPrediction,
    confidence,
    predictionHistory,
    modelLoaded,
    modelLoadingProgress,
    startDetection,
    stopDetection,
    startMediaPipeDetection,
    clearHistory
  };

  return (
    <SignLanguageContext.Provider value={value}>
      {children}
    </SignLanguageContext.Provider>
  );
};