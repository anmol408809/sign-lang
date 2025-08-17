@@ .. @@
   const processFrame = useCallback(async (canvas: HTMLCanvasElement) => {
     if (!detectorRef.current || !isDetecting || !modelLoaded) {
       return;
     }

     // Prevent concurrent processing
     if (processingRef.current) {
       return;
     }
     
-    // Throttle processing to 5 FPS for stability
+    // Throttle processing to 10 FPS for better responsiveness
     const now = Date.now();
-    if (now - lastProcessTimeRef.current < 200) {
+    if (now - lastProcessTimeRef.current < 100) {
       return;
     }
     
     processingRef.current = true;
     lastProcessTimeRef.current = now;

     try {
       const result = await detectorRef.current.detectGesture(canvas);
       
       if (!isDetecting) {
         processingRef.current = false;
         return;
       }
       
-      if (result && result.confidence > 0.5) {
+      if (result && result.confidence > 0.6) {
         console.log(`🎯 DETECTED: ${result.gesture} (${(result.confidence * 100).toFixed(1)}%)`);
         
         setCurrentPrediction(result.gesture);
         setConfidence(result.confidence);
         
         // Add to history if confidence is reasonable
-        if (result.confidence > 0.5) {
+        if (result.confidence > 0.6) {
           const newPrediction: Prediction = {
             gesture: result.gesture,
             confidence: result.confidence,
             timestamp: Date.now()
           };
           
           setPredictionHistory(prev => {
-            // Avoid duplicate consecutive predictions within 2 seconds
+            // Avoid duplicate consecutive predictions within 1 second
             const lastPrediction = prev[prev.length - 1];
             if (lastPrediction && 
                 lastPrediction.gesture === newPrediction.gesture && 
-                newPrediction.timestamp - lastPrediction.timestamp < 2000) {
+                newPrediction.timestamp - lastPrediction.timestamp < 1000) {
               return prev;
             }
             
             // Keep only last 50 predictions
             const updated = [...prev, newPrediction];
             return updated.slice(-50);
           });
         }
       } else {
-        // More gradual confidence reduction
-        setConfidence(prev => Math.max(0, prev * 0.95));
+        // Gradual confidence reduction
+        setConfidence(prev => Math.max(0, prev * 0.9));
         
         // Clear prediction if confidence drops below threshold
-        if (confidence < 0.4) {
+        if (confidence < 0.3) {
           setCurrentPrediction(null);
         }
       }
     } catch (error) {
       console.error('Error processing frame:', error);
     } finally {
       processingRef.current = false;
     }
   }, [isDetecting, modelLoaded, confidence]);