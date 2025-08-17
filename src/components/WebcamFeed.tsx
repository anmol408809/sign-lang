@@ .. @@
           // Draw video frame to canvas
           ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

+          // Draw a visual indicator that processing is happening
+          ctx.strokeStyle = '#22c55e';
+          ctx.lineWidth = 3;
+          ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
+
           // Process frame for hand detection
           processFrame(canvas);
         }
       } catch (error) {
         console.error('Error in detectFrame:', error);
       }

       if (isDetecting && cameraReady) {
         animationFrameRef.current = requestAnimationFrame(detectFrame);
       }
     };

     detectFrame();
   }, [isDetecting, cameraReady, processFrame]);