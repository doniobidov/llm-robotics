import subprocess
import time
import sys
import os
import signal

# Define the launch order
scripts = [
    ("Redis Reset", "reset.py"),
    ("Debug Visualizer", "debug_visualizer.py"),
    ("Tracker / SLAM Service", "tracker_service.py"),
    ("YOLO Camera Server", "camera_server.py"),
    ("Semantic Map Service", "camera_semantic_service.py"),
    ("Main Autonomy Loop", "main_autonomous.py")
]

def signal_handler(sig, frame):
    print("\n[Launch] Shutting down all robot services...")
    # Use pkill to find and terminate the specific python scripts running
    for _, script in scripts:
        os.system(f"pkill -f \"{script}\"")
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    print("=== Starting Go1 Autonomy Stack in Separate Terminals ===")
    
    for name, script in scripts:
        if not os.path.exists(script):
            print(f"[Warning] {script} not found in current directory! Skipping.")
            continue
            
        print(f"[Launch] Starting {name} ({script}) in a new terminal...")
        
        # Build the bash command safely handling spaces in paths
        # 'exec bash' ensures the terminal window stays open even if the script crashes or finishes
        bash_cmd = f'"{sys.executable}" "{script}"; echo ""; echo "--- Process Ended ---"; exec bash'
        
        # Spawn a new gnome-terminal window
        cmd = [
            "gnome-terminal", 
            f"--title={name}", 
            "--", 
            "bash", 
            "-c", 
            bash_cmd
        ]
        
        subprocess.Popen(cmd)
        
        # Give services a moment to initialize before starting the next one
        if script == "reset.py":
            time.sleep(1.0)  # Reset finishes instantly
        elif script in ["tracker_service.py", "camera_server.py"]:
            time.sleep(3.0)  # Tracker and Camera need more boot time
        else:
            time.sleep(1.0)
            
    print("\n=== All services launched! ===")
    print("=== Press Ctrl+C in THIS window to shut down all terminals safely. ===")
    
    # Keep the main thread alive to catch Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handled by signal_handler
        pass

if __name__ == "__main__":
    main()
