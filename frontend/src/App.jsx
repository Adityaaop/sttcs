import React, { useState, useEffect, useRef } from 'react';
import { Activity, AlertTriangle, Zap, Train, Play, Pause, AlertCircle } from 'lucide-react';
import './index.css';

const API_HOST = window.location.hostname || 'localhost';
const WEBSOCKET_URL = `ws://${API_HOST}:8000/ws/stream`;
const API_BASE_URL = `http://${API_HOST}:8000`;

export default function App() {
  const [mode, setMode] = useState('AI');
  const [ws, setWs] = useState(null);
  const [gameState, setGameState] = useState({
    trains: {},
    topology: { nodes: {}, edges: [] }
  });
  const [metrics, setMetrics] = useState({
    throughput: 0,
    avg_delay: 0,
    safety_violations: 0,
    interventions: 0
  });
  
  const [overrides, setOverrides] = useState({});
  const [activeDisruptions, setActiveDisruptions] = useState([]);
  
  const canvasRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket(WEBSOCKET_URL);
    
    socket.onopen = () => console.log("Connected to Stream");
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "state_update") {
        setGameState(data.state);
        setMetrics(data.metrics);
      }
    };
    
    socket.onclose = () => console.log("Disconnected from Stream");
    
    setWs(socket);
    
    return () => socket.close();
  }, []);

  // Draw schematic when state updates
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const stations = 4;
    const padding = 50;
    const trackLength = (canvas.width - padding * 2) / (stations - 1);
    
    const centerY = canvas.height / 2;
    
    // Draw nodes and edges
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 4;
    
    // Draw main lines
    ctx.beginPath();
    ctx.moveTo(padding, centerY);
    ctx.lineTo(canvas.width - padding, centerY);
    ctx.stroke();
    
    // Draw loop lines
    for(let i=0; i<stations; i++) {
      const x = padding + i * trackLength;
      
      // Draw loop line
      ctx.beginPath();
      ctx.moveTo(x - 20, centerY);
      ctx.quadraticCurveTo(x, centerY - 40, x + 20, centerY - 40);
      ctx.lineTo(x + trackLength/3, centerY - 40);
      ctx.quadraticCurveTo(x + trackLength/3 + 20, centerY - 40, x + trackLength/3 + 40, centerY);
      ctx.stroke();
      
      // Draw Station marker
      ctx.fillStyle = '#2a2a2a';
      ctx.fillRect(x - 30, centerY - 15, 60, 30);
      ctx.strokeStyle = '#555';
      ctx.strokeRect(x - 30, centerY - 15, 60, 30);
      
      ctx.fillStyle = '#fff';
      ctx.font = '12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(`STN ${i}`, x, centerY + 4);
    }
    
    // Draw Trains
    Object.values(gameState.trains).forEach(train => {
      // Find approximate position
      let x = padding;
      let y = centerY;
      
      // Parse track ID (e.g. Track_0_1_Fwd, Station_1, Loop_1)
      if (train.current_track.startsWith('Track_')) {
        const parts = train.current_track.split('_');
        const startStn = parseInt(parts[1]);
        const endStn = parseInt(parts[2]);
        const isFwd = parts[3] === 'Fwd';
        
        const startX = padding + startStn * trackLength;
        const progress = train.position / 10.0; // max length 10
        
        x = startX + progress * trackLength;
        y = isFwd ? centerY - 5 : centerY + 5;
      } else if (train.current_track.startsWith('Station_')) {
        const stn = parseInt(train.current_track.split('_')[1]);
        x = padding + stn * trackLength;
      } else if (train.current_track.startsWith('Loop_')) {
        const stn = parseInt(train.current_track.split('_')[1]);
        x = padding + stn * trackLength + 10;
        y = centerY - 40;
      }
      
      // Draw Train Rect
      ctx.fillStyle = train.priority === 1 ? '#ff8c00' : '#1890ff';
      ctx.fillRect(x - 10, y - 6, 20, 12);
      
      // Train ID
      ctx.fillStyle = '#fff';
      ctx.font = '10px Inter';
      ctx.fillText(train.id, x, y - 10);
      
      // Speed
      ctx.fillText(`${Math.round(train.speed)}`, x, y + 16);
    });
    
  }, [gameState]);

  const handleSpeedChange = (trainId, speed) => {
    setOverrides(prev => ({
      ...prev,
      [trainId]: { ...prev[trainId], speed: parseFloat(speed) }
    }));
    
    if (ws && mode === 'Manual') {
      ws.send(JSON.stringify({
        type: 'manual_override',
        train_id: trainId,
        speed: parseFloat(speed),
        signal_state: overrides[trainId]?.signal_state || 2
      }));
    }
  };
  
  const handleSignalChange = (trainId, signalState) => {
    setOverrides(prev => ({
      ...prev,
      [trainId]: { ...prev[trainId], signal_state: signalState }
    }));
    
    if (ws && mode === 'Manual') {
      ws.send(JSON.stringify({
        type: 'manual_override',
        train_id: trainId,
        speed: overrides[trainId]?.speed || 80,
        signal_state: signalState
      }));
    }
  };
  
  const handleDisruption = async (type, location) => {
    try {
      const isActive = activeDisruptions.includes(location);
      
      if (isActive) {
        await fetch(`${API_BASE_URL}/disruption/remove`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type, location })
        });
        setActiveDisruptions(prev => prev.filter(l => l !== location));
      } else {
        await fetch(`${API_BASE_URL}/disruption/inject`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type, location, duration_steps: 100, severity: 0.5 })
        });
        setActiveDisruptions(prev => [...prev, location]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="cockpit-container">
      <header className="header">
        <div className="header-title">
          <Activity color="#ff8c00" />
          <span>STTCS Cockpit</span>
        </div>
        
        <div className="mode-toggle">
          <button 
            className={mode === 'AI' ? 'active' : ''} 
            onClick={() => setMode('AI')}
          >
            Autonomous AI
          </button>
          <button 
            className={mode === 'Manual' ? 'active' : ''} 
            onClick={() => setMode('Manual')}
          >
            Manual Override
          </button>
        </div>
      </header>

      <section className="telemetry-strip">
        <div className="metric-card">
          <div className="metric-label">Section Throughput</div>
          <div className="metric-value">
            {metrics.throughput} <span className="metric-unit">trains/hr</span>
            <span className="metric-trend trend-up">+4.2%</span>
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Average Delay</div>
          <div className="metric-value">
            {metrics.avg_delay} <span className="metric-unit">min</span>
            <span className="metric-trend trend-down">-1.5m</span>
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Failsafe Interventions</div>
          <div className="metric-value" style={{ color: metrics.interventions > 0 ? 'var(--accent-red)' : 'var(--text-main)'}}>
            {metrics.interventions} <span className="metric-unit">events</span>
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Safety Violations</div>
          <div className="metric-value" style={{ color: 'var(--accent-green)'}}>
            {metrics.safety_violations} <span className="metric-unit">critical</span>
          </div>
        </div>
      </section>

      <main className="main-content">
        <div className="card schematic-container">
          <div className="card-title">Live Track Schematic</div>
          <canvas 
            ref={canvasRef} 
            className="track-schematic"
            width={1000}
            height={300}
          />
        </div>
        
        <div className="controls-section">
          <div className="card">
            <div className="card-title">Disruption Scenarios</div>
            <div className="control-group">
              <button 
                className={`disruption-btn ${activeDisruptions.includes('Station_1') ? 'active' : ''}`}
                onClick={() => handleDisruption('track_blockage', 'Station_1')}
              >
                <span>Track Blockage at Station 1</span>
                <AlertTriangle size={16} />
              </button>
              <button 
                className={`disruption-btn ${activeDisruptions.includes('Track_0_1_Fwd') ? 'active' : ''}`}
                onClick={() => handleDisruption('speed_restriction', 'Track_0_1_Fwd')}
              >
                <span>TSR (30km/h) Track 0-1</span>
                <AlertCircle size={16} />
              </button>
              <button 
                className={`disruption-btn ${activeDisruptions.includes('Station_2') ? 'active' : ''}`}
                onClick={() => handleDisruption('signal_failure', 'Station_2')}
              >
                <span>Signal Failure at Station 2</span>
                <Zap size={16} />
              </button>
            </div>
          </div>
          
          <div className="card" style={{ flex: 1 }}>
            <div className="card-title">Manual Dispatcher</div>
            {mode === 'AI' ? (
              <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '20px' }}>
                System is in Autonomous Mode.<br/>Switch to Manual to override.
              </div>
            ) : (
              <div className="control-group">
                {Object.values(gameState.trains).map(train => (
                  <div key={train.id} className="train-control">
                    <div className="train-header">
                      <div className="train-id"><Train size={14} style={{marginRight: '6px'}}/>{train.id}</div>
                      <div className={`train-priority ${train.priority === 1 ? 'high' : ''}`}>
                        Priority {train.priority}
                      </div>
                    </div>
                    
                    <div className="speed-slider-container">
                      <span style={{fontSize: '12px', color: 'var(--text-muted)'}}>SPD</span>
                      <input 
                        type="range" 
                        min="0" max="160" step="10"
                        className="speed-slider"
                        value={overrides[train.id]?.speed ?? train.speed}
                        onChange={(e) => handleSpeedChange(train.id, e.target.value)}
                      />
                      <span className="speed-value">{overrides[train.id]?.speed ?? Math.round(train.speed)}</span>
                    </div>
                    
                    <div className="signal-override">
                      <button 
                        className={`signal-btn signal-red ${(overrides[train.id]?.signal_state ?? 2) === 0 ? 'active' : ''}`}
                        onClick={() => handleSignalChange(train.id, 0)}
                      >RED</button>
                      <button 
                        className={`signal-btn signal-yellow ${(overrides[train.id]?.signal_state ?? 2) === 1 ? 'active' : ''}`}
                        onClick={() => handleSignalChange(train.id, 1)}
                      >YLW</button>
                      <button 
                        className={`signal-btn signal-green ${(overrides[train.id]?.signal_state ?? 2) === 2 ? 'active' : ''}`}
                        onClick={() => handleSignalChange(train.id, 2)}
                      >GRN</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
