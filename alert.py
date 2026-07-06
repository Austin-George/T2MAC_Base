import os, time, requests

BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_CHAT_ID_HERE"
RECEIVER_IP = "YOUR_LOCAL_PC_OR_SERVER_IP_HERE"

# Lightning AI folder path
FOLDER = os.path.expanduser("~/t2mac_results/models")
target = 500000
final_step = 5000000

def get_max_step():
    max_s = 0
    if not os.path.exists(FOLDER): return 0
    for root, dirs, files in os.walk(FOLDER):
        for d in dirs:
            if d.isdigit():
                max_s = max(max_s, int(d))
    return max_s

print("Monitoring started...")
while True:
    try:
        max_step = get_max_step()
        
        if max_step >= target:
            msg = f"🚀 T2MAC Update: Training has reached {max_step} steps!"
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg})
            target += 500000
        
        if max_step >= final_step:
            msg = f"✅ TRAINING COMPLETE ({max_step} steps)! Zipping data..."
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg})
            
            os.system("cd ~/ && tar -czvf t2mac_final_results.tar.gz t2mac_results")
            
            msg = f"☁️ Zipping complete! Transferring data safely back home..."
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg})
            
            # --- FILE TRANSFER OPTIONS ---
            # Option 1: SCP to your Local PC or Secure Server (Requires SSH Key setup)
            scp_cmd = (
                f"scp -o StrictHostKeyChecking=no -i ~/.ssh/your_private_key "
                f"~/t2mac_final_results.tar.gz user@{RECEIVER_IP}:~/"
            )
            os.system(scp_cmd)
            
            # Option 2: Alternatively, you could use 'rclone' to push to Google Drive:
            # os.system("rclone copy ~/t2mac_final_results.tar.gz gdrive:T2MAC_Backups/")
            
            msg = f"📦 Transfer complete! File is safely off the cloud. Shutting down Lightning Studio now to save money!"
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg})
            
            # This instantly shuts off the Lightning Studio VM
            os.system("sudo shutdown -h now")
            break
    except Exception as e:
        pass
    
    time.sleep(60)
