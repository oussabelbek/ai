import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import win32gui
import win32con
import win32api
import ctypes
from ctypes import wintypes
from PIL import Image, ImageGrab 
import keyboard
import os
import sys
import pytesseract
import unicodedata 
import re          

try:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
    pass 
except Exception as e:
    print(f"AVERTISSEMENT: Tesseract non configuré: {e}")

def press_key(vk_code):
    win32api.keybd_event(vk_code, 0, 0, 0) 
    time.sleep(0.03) # Augmenté légèrement pour voir si ça aide
    win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  

def left_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.02)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def right_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.02)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

class GameEnvironment:
    def __init__(self):
        self.screen_width = 1920 
        self.screen_height = 1080 
        self.action_space = 12
        self.observation_space = (84, 84, 3)
        self.game_window = None
        
        self.ai_player_name = "SPYMENDER" # REMPLACEZ PAR LE NOM DE VOTRE JOUEUR IA
        roi_left = 50; roi_top = 350; roi_width = 450; roi_height = 200 # EXEMPLE, À AJUSTER
        # Pillow ImageGrab.grab(bbox=...) attend (left, top, right, bottom)
        self.kill_feed_roi = (roi_left, roi_top, roi_left + roi_width, roi_top + roi_height) 
        
        self.last_event_time = 0 # Renommé pour gérer kill et mort
        self.last_event_killer = "" 
        self.last_event_victim = ""

        self.actions = {
            0: self.move_forward, 1: self.move_backward, 2: self.move_left,
            3: self.move_right, 4: self.look_left, 5: self.look_right,
            6: self.look_up, 7: self.look_down, 8: self.shoot,
            9: self.reload, 10: self.aim, 11: self.jump 
        }
        self.check_permissions()
        self.find_game_window()

    def check_permissions(self):
        print("Vérification des permissions...")
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
            print("Permissions administrateur détectées." if is_admin else "ATTENTION: Permissions administrateur recommandées.")
            print("Vérification des permissions admin terminée.")
        except Exception as e:
            print(f"Erreur lors de la vérification des permissions admin: {e}")
    
    def find_game_window(self):
        print("Recherche de la fenêtre de jeu...")
        def enum_windows_callback(hwnd, windows_list_param):
            is_visible = False
            try: is_visible = win32gui.IsWindowVisible(hwnd)
            except Exception: pass
            if is_visible:
                window_title = ""; 
                try: window_title = win32gui.GetWindowText(hwnd)
                except Exception: pass
                if window_title: 
                    title_lower = window_title.lower()
                    normalized_title = unicodedata.normalize('NFKD', title_lower)
                    cleaned_title = "".join(c for c in normalized_title if c.isalnum() or c.isspace() or c == '®')
                    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
                    # Décommentez pour voir tous les titres nettoyés:
                    # print(f"    Titre original: '{window_title}' -> Titre nettoyé: '{cleaned_title}' (HWND: {hwnd})") 
                    
                    keyword_to_match_clean = "call of duty®" 
                    if keyword_to_match_clean in cleaned_title:
                        print(f"        >>> CORRESPONDANCE TROUVÉE: '{window_title}' (nettoyé: '{cleaned_title}') correspond à '{keyword_to_match_clean}'") 
                        windows_list_param.append((hwnd, window_title))
                    else:
                        fallback_keywords = ['cod', 'warzone', 'modern warfare'] # Enlevez 'cod' si ça cause des faux positifs
                        for fk in fallback_keywords:
                            if fk in cleaned_title:
                                print(f"        >>> CORRESPONDANCE (fallback) TROUVÉE: '{window_title}' (nettoyé: '{cleaned_title}') correspond à '{fk}'") 
                                windows_list_param.append((hwnd, window_title))
                                break 
            return True
        found_windows_list = [] 
        try: win32gui.EnumWindows(enum_windows_callback, found_windows_list)
        except Exception as e: print(f"ERREUR EnumWindows: {e}")
        print("Fin de la recherche des fenêtres.")
        if found_windows_list:
            self.game_window = found_windows_list[0][0] 
            print(f"Fenêtre de jeu sélectionnée: {found_windows_list[0][1]} (HWND: {self.game_window})")
            self._activate_game_window()
        else:
            print("Fenêtre de Call of Duty non trouvée."); self.game_window = None

    def _activate_game_window(self):
        if not self.game_window or not win32gui.IsWindow(self.game_window): return False
        try:
            win32gui.ShowWindow(self.game_window, win32con.SW_RESTORE) 
            win32gui.SetForegroundWindow(self.game_window); 
            time.sleep(0.1); return win32gui.GetForegroundWindow() == self.game_window
        except Exception: return False

    def get_screen(self):
        bbox = None
        if self.game_window and win32gui.IsWindow(self.game_window):
            try:
                x, y, x2, y2 = win32gui.GetWindowRect(self.game_window)
                if x2 > x and y2 > y: bbox = (x, y, x2, y2)
            except Exception: pass 
        try: screenshot = ImageGrab.grab(bbox=bbox) 
        except Exception as e:
            print(f"ERREUR ImageGrab.grab: {e}. Image noire."); arr = np.zeros((self.observation_space[0], self.observation_space[1], 3), dtype=np.uint8)
            return cv2.resize(arr, (self.observation_space[1], self.observation_space[0]))
        arr = np.array(screenshot)
        if arr.ndim == 2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4: arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (self.observation_space[1], self.observation_space[0]))
    
    def preprocess_frame(self, frame_bgr): 
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) 
        return (gray.astype(np.float32) / 255.0)
        
    def process_kill_feed_event(self):
        """ Vérifie le kill feed. Retourne: 1 si IA kill, -1 si IA mort, 0 sinon. """
        try:
            img = ImageGrab.grab(bbox=self.kill_feed_roi)
            gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            # Le seuillage doit être ajusté en fonction des couleurs du texte vs fond
            # Si texte clair (jaune/rouge) sur fond variable/sombre:
            _, thresh_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY) 
            # cv2.imshow("OCR_Thresh", thresh_img); cv2.waitKey(1) # Pour déboguer le seuillage

            text = pytesseract.image_to_string(thresh_img, config=r'--oem 3 --psm 6 -l fra+eng')
            now = time.time()

            for line in reversed(text.splitlines()): # Traiter les lignes plus récentes en premier
                line_cleaned = line.strip().lower()
                if not line_cleaned: continue
                
                # print(f"DEBUG OCR Ligne: '{line_cleaned}'") # Décommentez pour voir chaque ligne
                parts = line_cleaned.split()
                
                if len(parts) >= 2: 
                    # Suppose: TUEUR [potentielle icone arme] VICTIME
                    # L'icône d'arme peut être mal interprétée ou être plusieurs "mots" pour l'OCR
                    # Donc on prend le premier mot comme tueur, le dernier comme victime.
                    # Ceci est une heuristique et peut nécessiter un affinage.
                    ocr_killer = parts[0]
                    ocr_victim = parts[-1]
                    
                    # Éviter de traiter le même événement exact plusieurs fois rapidement
                    if (ocr_killer == self.last_event_killer and \
                        ocr_victim == self.last_event_victim and \
                        now - self.last_event_time < 3.0): # Délai de 3s anti-rebond
                        continue

                    ai_name_lower = self.ai_player_name.lower()

                    if ocr_killer == ai_name_lower and ocr_victim != ai_name_lower:
                        print(f"CONFIRMATION D'ÉLIMINATION PAR {self.ai_player_name} SUR {ocr_victim.upper()} !")
                        self.last_event_time = now
                        self.last_event_killer = ocr_killer
                        self.last_event_victim = ocr_victim
                        return 1 # L'IA a fait un kill
                    elif ocr_victim == ai_name_lower and ocr_killer != ai_name_lower:
                        print(f"MORT DÉTECTÉE : {self.ai_player_name} TUÉ PAR {ocr_killer.upper()} !")
                        self.last_event_time = now
                        self.last_event_killer = ocr_killer
                        self.last_event_victim = ocr_victim
                        return -1 # L'IA est morte
            
        except Exception as e: print(f"Erreur OCR kill feed: {e}") 
        return 0 # Aucun événement pertinent détecté

    def get_reward(self, prev_gray_uint8, curr_gray_uint8): 
        reward = 0.01 # Survie
        if prev_gray_uint8.shape == curr_gray_uint8.shape:
            diff = cv2.absdiff(prev_gray_uint8, curr_gray_uint8)
            mavg = np.mean(diff) / 255.0 
            reward += 0.05 if mavg > 0.01 else -0.03 if mavg < 0.001 else 0
        
        event_type = self.process_kill_feed_event()
        if event_type == 1: # L'IA a fait un kill
            reward += 50
        elif event_type == -1: # L'IA est morte
            reward -= 25 # Pénalité pour la mort
        
        return reward

    def execute_action(self, action_id): 
        if action_id in self.actions:
            if not self._activate_game_window(): pass 
            self.actions[action_id]()

    # --- AJOUTEZ VOS PRINT() DE DÉBOGAGE DANS CES MÉTHODES D'ACTION ---
    def move_forward(self):  print("DEBUG ACTION: Avancer"); press_key(0x57) 
    def move_backward(self): print("DEBUG ACTION: Reculer"); press_key(0x53) 
    def move_left(self):     print("DEBUG ACTION: Gauche"); press_key(0x41) 
    def move_right(self):    print("DEBUG ACTION: Droite"); press_key(0x44) 
    def look_left(self):     print("DEBUG ACTION: Regard Gauche"); win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -30, 0, 0, 0) 
    def look_right(self):    print("DEBUG ACTION: Regard Droite"); win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 30, 0, 0, 0)
    def look_up(self):       print("DEBUG ACTION: Regard Haut"); win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, -20, 0, 0)
    def look_down(self):     print("DEBUG ACTION: Regard Bas"); win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, 20, 0, 0)
    def shoot(self):         print("DEBUG ACTION: Tirer"); left_click()
    def reload(self):        print("DEBUG ACTION: Recharger"); press_key(0x52) 
    def aim(self):           print("DEBUG ACTION: Viser"); right_click()   
    def jump(self):          print("DEBUG ACTION: Sauter"); press_key(0x20)

    def get_player_action(self, prev_mouse_pos):
        if keyboard.is_pressed('w'): return 0
        if keyboard.is_pressed('s'): return 1
        if keyboard.is_pressed('a'): return 2
        if keyboard.is_pressed('d'): return 3
        if win32api.GetKeyState(win32con.VK_LBUTTON) < 0: return 8
        if keyboard.is_pressed('r'): return 9
        if win32api.GetKeyState(win32con.VK_RBUTTON) < 0: return 10
        if keyboard.is_pressed('space'): return 11
        curr=win32api.GetCursorPos();dx=curr[0]-prev_mouse_pos[0];dy=curr[1]-prev_mouse_pos[1]
        if dx<-15: return 4
        if dx>15:  return 5
        if dy<-10: return 6
        if dy>10:  return 7
        return None

class DQNNetwork(nn.Module):
    def __init__(self, input_c, h, w, n_actions):
        super().__init__(); self.conv = nn.Sequential(nn.Conv2d(input_c,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),nn.Conv2d(64,64,3,1),nn.ReLU())
        with torch.no_grad(): conv_out_size = self.conv(torch.zeros(1,input_c,h,w)).view(1,-1).size(1)
        self.fc = nn.Sequential(nn.Linear(conv_out_size,512),nn.ReLU(),nn.Linear(512,n_actions))
    def forward(self,x): return self.fc(self.conv(x).view(x.size(0),-1))

class DQNAgent:
    def __init__(self, input_c, h, w, n_actions, lr=1e-4, gamma=0.99, epsilon_decay=0.995):
        self.input_channels = input_c; self.height=h; self.width=w; self.n_actions=n_actions
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Utilisation: {self.device}")
        self.q_net=DQNNetwork(input_c,h,w,n_actions).to(self.device); self.target_net=DQNNetwork(input_c,h,w,n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict()); self.target_net.eval()
        self.optimizer=optim.Adam(self.q_net.parameters(),lr=lr); self.memory=deque(maxlen=10000)
        self.epsilon=1.0; self.epsilon_min=0.01; self.epsilon_decay_rate=epsilon_decay; self.gamma=gamma; self.batch_size=32
    def load_demo_data(self,path):
        if not os.path.exists(path):
            return 0
        try:
            data=np.load(path,allow_pickle=True)
        except Exception:
            return 0
        for tr in data:
            self.memory.append(tuple(tr))
        print(f"Démonstration chargée: {len(data)} transitions")
        return len(data)
    def act(self,s):
        if random.random()<self.epsilon:return random.randrange(self.n_actions)
        with torch.no_grad():return self.q_net(torch.FloatTensor(s).unsqueeze(0).to(self.device)).argmax().item()
    def remember(self, *args):self.memory.append(args)
    def replay(self):
        if len(self.memory)<self.batch_size:return 0.0
        s,a,r,ns,d=zip(*random.sample(self.memory,self.batch_size))
        s_t=torch.FloatTensor(np.array(s)).to(self.device);ns_t=torch.FloatTensor(np.array(ns)).to(self.device)
        a_t=torch.LongTensor(a).unsqueeze(1).to(self.device);r_t=torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d_t=torch.BoolTensor(d).unsqueeze(1).to(self.device)
        cq=self.q_net(s_t).gather(1,a_t);nq_target=self.target_net(ns_t).max(1)[0].unsqueeze(1)
        target_q=r_t+self.gamma*nq_target*~d_t;loss=nn.MSELoss()(cq,target_q)
        self.optimizer.zero_grad();loss.backward();self.optimizer.step()
        self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay_rate);return loss.item()
    def update_target_network_periodically(self,ep,freq):
        if ep%freq==0 and ep>0:self.target_net.load_state_dict(self.q_net.state_dict());print(f"Réseau Cible MAJ ep {ep}.")

class DemoRecorder:
    def __init__(self, env):
        self.env = env
        self.records = []
    def record(self, steps=1000, out_file="demo_data.npy"):
        prev_frame = self.env.preprocess_frame(self.env.get_screen())
        prev_mouse = win32api.GetCursorPos()
        for _ in range(steps):
            act = self.env.get_player_action(prev_mouse)
            time.sleep(0.05)
            curr_frame_raw = self.env.get_screen()
            curr_frame = self.env.preprocess_frame(curr_frame_raw)
            reward = self.env.get_reward((prev_frame*255).astype(np.uint8),(curr_frame*255).astype(np.uint8))
            done = False
            if act is not None:
                self.records.append((prev_frame, act, reward, curr_frame, done))
            prev_frame = curr_frame
            prev_mouse = win32api.GetCursorPos()
        np.save(out_file, np.array(self.records, dtype=object))
        print(f"Démonstration enregistrée : {out_file} ({len(self.records)} transitions)")

class CODTrainer:
    def __init__(self):
        self.env=GameEnvironment();h,w,c=self.env.observation_space
        self.agent=DQNAgent(c,h,w,self.env.action_space)
        self.stats={'episodes_completed':0,'best_episode_reward':-float('inf')}
        demo_file="demo_data.npy"
        if os.path.exists(demo_file):
            self.agent.load_demo_data(demo_file)
    def train(self,num_total_eps=1000):
        print(f"Début entraînement pour {num_total_eps} épisodes...");init_ep=self.stats.get('episodes_completed',0)
        for ep in range(init_ep,num_total_eps):
            ep_disp_num=ep+1;print(f"\n--- Début Épisode {ep_disp_num} ---");total_ep_r=0.0
            try:
                prev_raw_bgr=self.env.get_screen();prev_proc_frame=self.env.preprocess_frame(prev_raw_bgr)
                state_chw=np.stack([prev_proc_frame]*self.agent.input_channels,0)
            except Exception as e:print(f"Erreur init état: {e}");break
            for step in range(1000):
                action=self.agent.act(state_chw) 
                # print(f"Choix Action: {action}") # Le print DEBUG dans GameEnvironment est plus spécifique
                self.env.execute_action(action);time.sleep(0.05)
                curr_raw_bgr=self.env.get_screen();curr_proc_frame=self.env.preprocess_frame(curr_raw_bgr)
                reward=self.env.get_reward((prev_proc_frame*255).astype(np.uint8),(curr_proc_frame*255).astype(np.uint8))
                done=(step==999);next_state_chw=np.stack([curr_proc_frame]*self.agent.input_channels,0)
                self.agent.remember(state_chw,action,reward,next_state_chw,done);self.agent.replay()
                total_ep_r+=reward;state_chw=next_state_chw;prev_proc_frame=curr_proc_frame
                if keyboard.is_pressed('q'):print("Arrêt.");self.save_model();return
                if keyboard.is_pressed('p'):print("Pause. P pour reprendre.");keyboard.wait('p',suppress=True);time.sleep(0.2);keyboard.wait('p',suppress=True);print("Reprise.")
            print(f"Fin Ép {ep_disp_num}: R={total_ep_r:.2f}, Eps={self.agent.epsilon:.4f}");self.stats['episodes_completed']=ep_disp_num
            if total_ep_r>self.stats['best_episode_reward']:self.stats['best_episode_reward']=total_ep_r;print(f"Nouveau best reward: {total_ep_r:.2f}")
            self.agent.update_target_network_periodically(ep_disp_num,10)
            if ep_disp_num%20==0:self.save_model()
        self.save_model()
    def save_model(self,fn=None):
        if not os.path.exists("models"):os.makedirs("models")
        if fn is None:fn=f"cod_ai_ep{self.stats['episodes_completed']}.pth"
        fp=os.path.join("models",fn)
        try:torch.save({'q_network_state_dict':self.agent.q_net.state_dict(),'target_network_state_dict':self.agent.target_net.state_dict(),'optimizer_state_dict':self.agent.optimizer.state_dict(),'epsilon':self.agent.epsilon,'stats':self.stats},fp);print(f"Modèle sauvegardé: {fp}")
        except Exception as e:print(f"Erreur sauvegarde: {e}")
    def load_model(self,fn):
        if "models" not in fn.replace("\\","/"):fp=os.path.join("models",fn)
        else:fp=fn
        if not os.path.exists(fp):print(f"Modèle non trouvé: {fp}");return False
        try:
            ckpt=torch.load(fp,map_location=self.agent.device);self.agent.q_net.load_state_dict(ckpt['q_network_state_dict'])
            if 'target_network_state_dict' in ckpt:self.agent.target_net.load_state_dict(ckpt['target_network_state_dict'])
            else:self.agent.target_net.load_state_dict(ckpt['q_network_state_dict'])
            self.agent.optimizer.load_state_dict(ckpt['optimizer_state_dict']);self.agent.epsilon=ckpt.get('epsilon',1.0)
            s_stats=ckpt.get('stats');
            if s_stats:self.stats=s_stats
            else:
                try:self.stats['episodes_completed']=int(fn.split('_ep')[-1].split('.')[0])
                except:self.stats['episodes_completed']=0
            print(f"Modèle chargé: {fp}\n Eps: {self.agent.epsilon:.3f}, Ép complétés: {self.stats['episodes_completed']}")
            return True
        except Exception as e:print(f"Erreur chargement: {fp}: {e}");return False

if __name__=="__main__":
    print("=== IA Call of Duty Trainer (Win32 Input) ===");trainer=CODTrainer()
    choice=input("Nouveau (N), Charger (C) ou Demo (D) ? [N/C/D]:").strip().upper()
    if choice=='D':
        try:steps=int(input("Nombre d'etapes a enregistrer (1000 defaut):") or "1000")
        except ValueError:steps=1000
        recorder=DemoRecorder(trainer.env)
        print("Debut de l'enregistrement dans 3s...");time.sleep(3)
        recorder.record(steps)
        sys.exit(0)
    if choice=='C':
        mf=input("Fichier modèle (vide=dernier):").strip()
        if not mf:
            mdir="models"
            if os.path.exists(mdir):
                mfs=[f for f in os.listdir(mdir) if f.endswith('.pth')and f.startswith('cod_ai_ep')]
                if mfs:
                    try:mfs.sort(key=lambda n:int(n.split('_ep')[-1].split('.')[0]),reverse=True)
                    except ValueError:mfs.sort(reverse=True)
                    mf=mfs[0];print(f"Chargement dernier: {mf}")
                else:print("Aucun modèle. Nouvel entraînement.");choice='N'
            else:print("Dossier 'models' absent. Nouvel entraînement.");choice='N'
        if choice=='C' and not trainer.load_model(mf):print("Échec chargement. Nouvel entraînement.");choice='N'
    if choice=='N':
        print("Init nouvel entraînement.");trainer.stats={'episodes_completed':0,'best_episode_reward':-float('inf')}
        trainer.agent.epsilon=1.0;h,w,c=trainer.env.observation_space
        trainer.agent.q_net=DQNNetwork(c,h,w,trainer.env.action_space).to(trainer.agent.device)
        trainer.agent.target_net=DQNNetwork(c,h,w,trainer.env.action_space).to(trainer.agent.device)
        trainer.agent.target_net.load_state_dict(trainer.agent.q_net.state_dict())
        lr_val=trainer.agent.optimizer.defaults.get('lr',1e-4) if hasattr(trainer.agent.optimizer,'defaults') else 1e-4
        trainer.agent.optimizer=optim.Adam(trainer.agent.q_net.parameters(),lr=lr_val)
    try:
        n_eps=int(input(f"Nb épisodes CETTE SESSION (actuel: {trainer.stats.get('episodes_completed',0)}):")or"50")
        if n_eps<=0:raise ValueError
    except ValueError:print("Nb invalide, defaut 50.");n_eps=50
    total_target_eps=trainer.stats.get('episodes_completed',0)+n_eps
    print(f"\nEntraînement jusqu'à {total_target_eps} épisodes.");print("Début 5s... CoD au premier plan!")
    for i in range(5,0,-1):print(f"   {i}...");time.sleep(1)
    trainer.train(total_target_eps);print("Programme terminé.")