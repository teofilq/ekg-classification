# 📌 Proiect: Clasificare Aritmii ECG cu Un Singur Lead

## 🔥 Descriere
Acest proiect implementează un sistem simplificat pentru **clasificarea aritmiilor** folosind date ECG dintr-o singură derivație (de preferat **Lead II**). Se bazează pe un pipeline multi-etapă care include:
1. **Filtrare și curățare a semnalului** (preprocesare)
2. **Extragere de caracteristici** (feature engineering bazat pe dicionare numerice)
3. **Clasificare** cu algoritmi de machine learning (Random Forest, Gradient Boosting etc.)

Inspirat de studiul "Optimal Multi-Stage Arrhythmia Classification Approach" (Zheng et al.), această implementare simplificată încearcă să echilibreze **acuratețea** și **eficiența** pentru aplicații rapide și practice.

---

## 🚀 Etape principale

### 1️⃣ **Preprocesare și filtrare semnal**
Pentru a elimina zgomotul și a îmbunătăți calitatea datelor ECG:
- **Filtru trece-jos (Butterworth)** pentru eliminarea zgomotului de înaltă frecvență (>50Hz)
- **Filtrare baseline drift** (LOESS sau high-pass filter ~0.5Hz)
- **Filtru de netezire** (optional: median sau Non-Local Means)

Astfel, semnalul final este mai curat și mai stabil, facilitând extragerea corectă a caracteristicilor.

---

### 2️⃣ **Extragerea de caracteristici (Feature Extraction)**
Studiul introduce o metodă nouă, complexă, de extragere de caracteristici bazată pe:
1. **Măsurători ECG de bază** (furnizate de aparat): rata ventriculară, rata atrială, durata QRS, intervalul QT, axele QRS/T, etc.
2. **Statistici privind intervalele RR** (media, variația, numărul QRS) și parametri ai undelor (în special din derivația II): înălțime, lățime, „prominență” (prominence) a vârfurilor (peaks) și a văilor (valleys).
3. **Relația între vârfuri și văi**: raporturi între diferențe de înălțime/lățime/prominență și diferența de timp dintre ele. Pentru a putea fi folosite uniform, aceste raporturi sunt împărțite pe intervale și convertite în distribuții empirice (practic se construiește un „histogram-based feature” sub formă de „dicționar numeric” cu frecvențele valorilor).

Autoriii au testat **11 combinații de seturi de caracteristici** (de la un set minimal la unul foarte complex cu zeci de mii de features), apoi au rulat diverse clasificatoare cu hiperparametri ajustați exhaustiv (Grid Search).

---

### 3️⃣ **Clasificare (Modele și rezultate)**
Folosim algoritmi de **machine learning** pentru clasificarea aritmiilor în 4 clase principale:
- **Atrial Fibrillation (AFIB)**
- **General Supraventricular Tachycardia (GSVT)**
- **Sinus Bradycardia (SB)**
- **Sinus Rhythm (SR)**

🔹 **Modele testate:**
✅ Random Forest  
✅ XGBoost (Extreme Gradient Boosting)  
✅ SVM (Support Vector Machines)  
✅ MLP (Multi-layer Perceptron Neural Network)  

📊 **Metrici de evaluare:** F1-Score, Confusion Matrix, Acuratețe (Accuracy)

Concluzii-cheie din rezultate:
- **Cea mai bună performanță** pentru pacienții *fără* alte condiții cardiace a fost atinsă cu **Extreme Gradient Boosting Tree** (F1-Score = 0.988).
- Pentru pacienții *cu* condiții cardiace adiționale, **Gradient Boosting Tree** a funcționat cel mai bine (F1-Score = 0.97).
- *Rescalarea amplitudinii* semnalelor (astfel încât vârful maxim să fie 1) a adus un plus foarte mic, dar ușor pozitiv, la acuratețe.
- **Folosirea a 12 derivații crește performanța cu aproximativ 1-2% față de folosirea unei singure derivații (de obicei lead II).**
- Metoda propusă a obținut apoi **F1-Score = 0.992** și pe baza de date externă MIT-BIH.

---

## 🔬 Concluzii și Direcții Viitoare
✅ **Un singur lead este suficient** pentru clasificarea aritmiilor, dar cu o acuratețe ușor mai mică decât modelele pe 12-lead.
✅ **Gradient Boosting și Random Forest** sunt cele mai robuste modele pentru această sarcină.
✅ **Un pipeline simplificat** bazat pe dicionare numerice oferă rezultate competitive și este mai interpretabil decât rețelele neuronale complexe.

🔜 **Îmbunătățiri viitoare:**
- Testare pe date în timp real de la dispozitive purtabile
- Optimizare a caracteristicilor prin selecție automată
- Integrare cu API-uri pentru predicție rapidă în cloud

---
📌 **Autor:** [Numele Tău]  
📅 **Ultima actualizare:** [Data]

