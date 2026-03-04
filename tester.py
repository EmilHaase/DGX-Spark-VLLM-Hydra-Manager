import os
import time
import json
import urllib.request
import threading
from urllib.error import URLError

class TesterMenu:
    def __init__(self):
        self.active_models = []
        self.selected_idx = 0
        self.active = True
        
    def scan_ports(self):
        self.active_models.clear()
        
        # Ping standard vLLM ports
        for port in range(8000, 8008):
            url = f"http://127.0.0.1:{port}/v1/models"
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    data = json.loads(response.read().decode())
                    if data and "data" in data and len(data["data"]) > 0:
                        model_name = data["data"][0].get("id", "Unknown")
                        max_len = data["data"][0].get("max_model_len", "Unknown")
                        
                        self.active_models.append({
                            "port": port,
                            "name": model_name,
                            "ctx": max_len
                        })
            except (URLError, ConnectionResetError, json.JSONDecodeError):
                pass
                
        # Ensure selection is valid
        if self.active_models and self.selected_idx >= len(self.active_models):
            self.selected_idx = 0

    def draw(self):
        os.system("clear")
        print("=" * 80)
        print(" " * 22 + "DGX Spark VLLM Engine Benchmarks")
        print("=" * 80)
        
        if not self.active_models:
            print("\nServers Offline. No active vLLM engines detected on ports 8000-8007.")
            print("\nPlease return to Master Menu and launch engines first.")
        else:
            print(f"Detected {len(self.active_models)} running engines:\n")
            for i, model in enumerate(self.active_models):
                mark = "[X]" if i == self.selected_idx else "[ ]"
                color = "\033[96m" if i == self.selected_idx else "\033[0m"
                print(f"{color}{i+1:2d}) {mark} Port: {model['port']} | Model: {model['name']}{color} (Ctx: {model['ctx']})")
                
        print("-" * 80)
        print("Tests:")
        print(" l        : Latency Test (Time-To-First-Token)")
        print(" s        : Speed Test (Generation TPS)")
        print(" c        : Interactive Chat Shell (Streaming)")
        print(" p        : Parallel Load Test (Threaded stress test)")
        print("-" * 80)
        print("Commands:")
        print(" <ID>     : Select engine target (e.g., 1)")
        print(" scan     : Refresh port scanners")
        print(" q        : Quit to Master Menu\n")
        
    def handle_input(self, cmd: str):
        cmd = cmd.strip().lower()
        if not cmd:
            return
            
        if cmd == "q":
            self.active = False
        elif cmd == "scan":
            print("Scanning ports...")
            self.scan_ports()
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(self.active_models):
                self.selected_idx = idx
        elif cmd == "l":
            self.run_latency()
        elif cmd == "s":
            self.run_speed()
        elif cmd == "c":
            self.run_chat()
        elif cmd == "p":
            self.run_parallel()
            
    def _post(self, port: int, payload: dict, stream: bool = False):
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        payload["stream"] = stream
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req)

    def run_latency(self):
        if not self.active_models: return
        target = self.active_models[self.selected_idx]
        
        print(f"\n>> Running Latency Test (TTFT) on {target['name']}...")
        payload = {
            "model": target['name'],
            "messages": [{"role": "user", "content": "Reply with a simple 'Hello world!'."}],
            "max_tokens": 10
        }
        
        try:
            start = time.time()
            res = self._post(target['port'], payload, stream=True)
            ttft = None
            
            for line in res:
                if line and line.strip() != b"data: [DONE]":
                    ttft = time.time() - start
                    break
                    
            print(f"✅ Success! Time-To-First-Token: \033[92m{ttft:.3f} seconds\033[0m")
        except Exception as e:
            print(f"❌ Error: {e}")
        input("Press Enter to continue...")

    def run_speed(self):
        if not self.active_models: return
        target = self.active_models[self.selected_idx]
        
        print(f"\n>> Running Speed Test (TPS) on {target['name']}...")
        print("Generating a long response...")
        payload = {
            "model": target['name'],
            "messages": [{"role": "user", "content": "Write a 500 word story about a space pirate."}],
            "max_tokens": 800
        }
        
        try:
            start = time.time()
            with self._post(target['port'], payload, stream=False) as res:
                data = json.loads(res.read().decode())
                
                duration = time.time() - start
                total_tokens = data.get("usage", {}).get("completion_tokens", 0)
                
                if total_tokens > 0:
                    tps = total_tokens / duration
                    print(f"\nGenerated \033[96m{total_tokens}\033[0m tokens in \033[93m{duration:.2f}s\033[0m")
                    print(f"✅ Generation Speed: \033[92m{tps:.2f} tokens/second\033[0m")
                else:
                    print(f"❌ Completed but generated 0 tokens.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        input("Press Enter to continue...")

    def run_chat(self):
        if not self.active_models: return
        target = self.active_models[self.selected_idx]
        
        print("\n" + "=" * 50)
        print(f"Chat Mode: {target['name']}")
        print("Type your message. Press Ctrl+C to exit chat.")
        print("=" * 50)
        
        messages = []
        while True:
            try:
                user_msg = input("\n\033[96mUser>\033[0m ")
                if not user_msg.strip(): continue
                
                messages.append({"role": "user", "content": user_msg})
                
                payload = {
                    "model": target['name'],
                    "messages": messages,
                    "max_tokens": 4000
                }
                
                print("\033[92mBot> \033[0m", end="", flush=True)
                
                full_response = ""
                res = self._post(target['port'], payload, stream=True)
                for line in res:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk["choices"][0].get("delta", {}).get("content", "")
                            # Ignore <think> reasoning tags
                            if content and not content.startswith("<think>") and not content.startswith("</think>"):
                                full_response += content
                                print(content, end="", flush=True)
                        except:
                            pass
                print() # Newline
                messages.append({"role": "assistant", "content": full_response})
                
            except KeyboardInterrupt:
                print("\nExiting chat mode...")
                break
            except Exception as e:
                print(f"\n❌ API Error: {e}")
                
    def _parallel_worker(self, thread_id: int, target: dict, event: threading.Event, results: list):
        payload = {
            "model": target['name'],
            "messages": [{"role": "user", "content": f"Tell me a short joke {thread_id}."}],
            "max_tokens": 100
        }
        
        event.wait() # Wait for all threads to be ready to fire simultaneously
        start = time.time()
        
        try:
            with self._post(target['port'], payload, stream=False) as res:
                data = json.loads(res.read().decode())
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                dur = time.time() - start
                results[thread_id] = f"✅ Thread {thread_id}: {tokens} tokens in {dur:.2f}s ({(tokens/max(0.01, dur)):.1f} t/s)"
        except Exception as e:
            results[thread_id] = f"❌ Thread {thread_id}: Error - {str(e)[:40]}"

    def run_parallel(self):
        if not self.active_models: return
        target = self.active_models[self.selected_idx]
        
        try:
            count_str = input("\nEnter number of parallel requests to send simultaneously: ").strip()
            num_reqs = int(count_str)
            if num_reqs <= 0 or num_reqs > 500:
                print("Please enter a valid number (1-500).")
                return
        except ValueError:
            return
            
        print(f"\n>> Spawning {num_reqs} parallel threads against {target['name']}...")
        
        results = ["⏳ Waiting..." for _ in range(num_reqs)]
        threads = []
        sync_event = threading.Event()
        
        for i in range(num_reqs):
            t = threading.Thread(target=self._parallel_worker, args=(i, target, sync_event, results))
            t.daemon = True
            threads.append(t)
            t.start()
            
        print(f"Firing {num_reqs} requests simultaneously... \033[93m(Press Ctrl+C to abort)\033[0m\n")
        sync_event.set() # Release the hounds
        
        try:
            # Live updating dashboard
            overall_start = time.time()
            all_done = False
            
            while not all_done:
                # Move cursor up
                if time.time() - overall_start > 0.1:
                    print(f"\033[{num_reqs}A", end="")
                    
                for r in results:
                    print(f"\033[K{r}") # Clear line and print
                    
                all_done = not any(t.is_alive() for t in threads)
                time.sleep(0.1)
                
            print(f"\nParallel test completed in {time.time() - overall_start:.2f}s.")
        except KeyboardInterrupt:
            print("\nParallel test aborted by user. Some threads may still finish in background.")
            
        input("\nPress Enter to continue...")

    def run(self):
        self.scan_ports()
        while self.active:
            self.draw()
            try:
                cmd = input("test> ")
                self.handle_input(cmd)
            except (KeyboardInterrupt, EOFError):
                self.active = False
