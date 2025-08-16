export interface LandmarkSequence {
  landmarks: number[][];
  label: string;
}

export class DataProcessor {
  private readonly SEQUENCE_LENGTH = 30;
  private readonly LANDMARK_COUNT = 42; // 21 landmarks * 2 coordinates (x, y)

  /**
   * Simulates processing video files to extract landmark sequences
   * In a real implementation, this would use MediaPipe to extract hand landmarks from videos
   */
  async processVideoData(): Promise<LandmarkSequence[]> {
    console.log('🎬 Processing video data...');
    
    const sequences: LandmarkSequence[] = [];
    
    // Simulate processing hello videos (5 videos)
    for (let i = 1; i <= 5; i++) {
      const helloSequence = this.generateRealisticSequence('hello', i);
      sequences.push({
        landmarks: helloSequence,
        label: 'hello'
      });
    }
    
    // Simulate processing thankyou videos (5 videos)
    for (let i = 1; i <= 5; i++) {
      const thankyouSequence = this.generateRealisticSequence('thankyou', i);
      sequences.push({
        landmarks: thankyouSequence,
        label: 'thankyou'
      });
    }
    
    console.log(`✅ Processed ${sequences.length} video sequences`);
    return sequences;
  }

  /**
   * Generates realistic hand landmark sequences for training
   * This simulates what would be extracted from actual video files
   */
  private generateRealisticSequence(gesture: string, videoIndex: number): number[][] {
    const sequence: number[][] = [];
    
    for (let frame = 0; frame < this.SEQUENCE_LENGTH; frame++) {
      const landmarks: number[] = [];
      const progress = frame / this.SEQUENCE_LENGTH;
      
      // Generate 21 hand landmarks (x, y coordinates each)
      for (let i = 0; i < 21; i++) {
        let x, y;
        
        if (gesture === 'hello') {
          // Hello gesture: hand moves in a waving pattern
          const wavePhase = progress * Math.PI * 2 + (videoIndex * 0.5);
          const baseX = 0.5 + Math.sin(wavePhase) * 0.1;
          const baseY = 0.4 + Math.cos(wavePhase * 0.5) * 0.05;
          
          // Different landmarks have different positions relative to palm
          const fingerOffset = this.getFingerOffset(i);
          x = baseX + fingerOffset.x + Math.sin(wavePhase + i * 0.1) * 0.02;
          y = baseY + fingerOffset.y + Math.cos(wavePhase + i * 0.1) * 0.02;
          
        } else { // thankyou
          // Thank you gesture: more stable, slight forward movement
          const baseX = 0.5 + Math.sin(progress * Math.PI + videoIndex) * 0.03;
          const baseY = 0.45 + progress * 0.05; // Slight forward movement
          
          const fingerOffset = this.getFingerOffset(i);
          x = baseX + fingerOffset.x + Math.sin(progress * Math.PI * 0.5 + i * 0.2) * 0.01;
          y = baseY + fingerOffset.y + Math.cos(progress * Math.PI * 0.3 + i * 0.2) * 0.01;
        }
        
        // Add some realistic noise and clamp values
        x += (Math.random() - 0.5) * 0.02;
        y += (Math.random() - 0.5) * 0.02;
        
        landmarks.push(Math.max(0, Math.min(1, x)));
        landmarks.push(Math.max(0, Math.min(1, y)));
      }
      
      sequence.push(landmarks);
    }
    
    return sequence;
  }

  /**
   * Returns relative position offset for each hand landmark
   */
  private getFingerOffset(landmarkIndex: number): { x: number; y: number } {
    // Simplified hand landmark positions relative to palm center
    const offsets = [
      { x: 0, y: 0 },      // 0: Wrist
      { x: -0.02, y: -0.03 }, // 1: Thumb CMC
      { x: -0.04, y: -0.06 }, // 2: Thumb MCP
      { x: -0.06, y: -0.08 }, // 3: Thumb IP
      { x: -0.08, y: -0.10 }, // 4: Thumb TIP
      { x: 0.02, y: -0.08 },  // 5: Index MCP
      { x: 0.03, y: -0.12 },  // 6: Index PIP
      { x: 0.04, y: -0.15 },  // 7: Index DIP
      { x: 0.05, y: -0.18 },  // 8: Index TIP
      { x: 0.06, y: -0.06 },  // 9: Middle MCP
      { x: 0.08, y: -0.12 },  // 10: Middle PIP
      { x: 0.09, y: -0.16 },  // 11: Middle DIP
      { x: 0.10, y: -0.20 },  // 12: Middle TIP
      { x: 0.08, y: -0.04 },  // 13: Ring MCP
      { x: 0.09, y: -0.08 },  // 14: Ring PIP
      { x: 0.10, y: -0.12 },  // 15: Ring DIP
      { x: 0.11, y: -0.15 },  // 16: Ring TIP
      { x: 0.06, y: -0.01 },  // 17: Pinky MCP
      { x: 0.07, y: -0.04 },  // 18: Pinky PIP
      { x: 0.08, y: -0.07 },  // 19: Pinky DIP
      { x: 0.09, y: -0.10 }   // 20: Pinky TIP
    ];
    
    return offsets[landmarkIndex] || { x: 0, y: 0 };
  }

  /**
   * Standardizes sequences to fixed length
   */
  standardizeSequence(sequence: number[][], targetLength: number = this.SEQUENCE_LENGTH): number[][] {
    if (sequence.length === targetLength) {
      return sequence;
    }
    
    if (sequence.length > targetLength) {
      // Truncate to target length
      return sequence.slice(0, targetLength);
    } else {
      // Pad with zeros
      const padded = [...sequence];
      const zeroFrame = new Array(this.LANDMARK_COUNT).fill(0);
      
      while (padded.length < targetLength) {
        padded.push([...zeroFrame]);
      }
      
      return padded;
    }
  }

  /**
   * Converts label strings to numeric values
   */
  encodeLabels(sequences: LandmarkSequence[]): { sequences: number[][][]; labels: number[] } {
    const labelMap: { [key: string]: number } = {
      'hello': 0,
      'thankyou': 1
    };

    const processedSequences = sequences.map(seq => 
      this.standardizeSequence(seq.landmarks)
    );

    const labels = sequences.map(seq => labelMap[seq.label]);

    return {
      sequences: processedSequences,
      labels: labels
    };
  }

  /**
   * Augments training data by creating variations
   */
  augmentData(sequences: LandmarkSequence[]): LandmarkSequence[] {
    const augmented: LandmarkSequence[] = [...sequences];
    
    // Create variations for each sequence
    sequences.forEach(seq => {
      // Add noise variation
      const noisySequence = seq.landmarks.map(frame => 
        frame.map(coord => coord + (Math.random() - 0.5) * 0.02)
      );
      augmented.push({
        landmarks: noisySequence,
        label: seq.label
      });
      
      // Add scale variation
      const scaledSequence = seq.landmarks.map(frame => 
        frame.map(coord => coord * (0.9 + Math.random() * 0.2))
      );
      augmented.push({
        landmarks: scaledSequence,
        label: seq.label
      });
    });
    
    console.log(`📈 Data augmented: ${sequences.length} → ${augmented.length} sequences`);
    return augmented;
  }
}