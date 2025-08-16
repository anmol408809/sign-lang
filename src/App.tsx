@@ .. @@
 import React from 'react';
 import Header from './components/Header';
 import Hero from './components/Hero';
 import DetectionInterface from './components/DetectionInterface';
+import TrainingInterface from './components/TrainingInterface';
 import Features from './components/Features';
 import Footer from './components/Footer';
 import { SignLanguageProvider } from './context/SignLanguageContext';

 function App() {
   return (
     <SignLanguageProvider>
       <div className="min-h-screen">
         <Header />
         <main>
           <Hero />
           <DetectionInterface />
+          <section id="training" className="py-20 bg-gradient-to-b from-gray-50/50 to-white">
+            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
+              <div className="text-center mb-12">
+                <h2 className="text-4xl font-bold text-gray-900 mb-4">
+                  Train Your Own Model
+                </h2>
+                <p className="text-xl text-gray-600 max-w-3xl mx-auto">
+                  Use the video data to train a custom LSTM model for sign language detection.
+                </p>
+              </div>
+              <TrainingInterface />
+            </div>
+          </section>
           <Features />
         </main>
         <Footer />
       </div>
     </SignLanguageProvider>
   );
 }

 export default App;