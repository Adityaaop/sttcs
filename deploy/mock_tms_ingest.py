import time
import json
import random
import requests
import asyncio
from typing import Dict, Any

API_URL = "http://localhost:8000"

class MockTMSWorker:
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.trains = ["T_Rajdhani_01", "T_Freight_02", "T_Express_03"]
        self.tracks = ["Track_0_1_Fwd", "Track_1_2_Fwd", "Track_2_3_Fwd"]
        
    def generate_telemetry_batch(self):
        # Simulate noisy/stochastic SCADA telemetry
        telemetry = {}
        for idx, t_id in enumerate(self.trains):
            telemetry[t_id] = {
                "speed_advisory": float(random.uniform(30.0, 130.0)),
                "signal_state": random.choice([0, 1, 2])
            }
        return telemetry
        
    def push_telemetry(self):
        while True:
            try:
                actions = self.generate_telemetry_batch()
                # Assuming /simulate/step accepts actions to step the environment
                response = requests.post(f"{self.api_url}/simulate/step", json={"actions": actions})
                if response.status_code == 200:
                    print(f"[TMS Ingest] Successfully pushed telemetry for {len(actions)} trains.")
                else:
                    print(f"[TMS Ingest] Error: HTTP {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"[TMS Ingest] Connection Failed: {e}")
                
            time.sleep(1.0) # 1Hz update loop
            
import os

if __name__ == "__main__":
    print("Starting Mock TMS Ingest Worker...")
    api_url = os.getenv("API_URL", "http://localhost:8000")
    worker = MockTMSWorker(api_url=api_url)
    worker.push_telemetry()
