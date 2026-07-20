# AETHER online — så din veninde kan prøve uden at installere noget

Tre realistiske niveauer. Vælg efter hastværk vs. “rigtig” deling.

---

## TL;DR — hvad skal du vælge?

| Situation | Løsning | Tid | Til dig | Til hende |
|-----------|---------|-----|---------|-----------|
| **I aften, 1 veninde, 30 min** | **A) Tunnel** (din PC kører appen) | ~5 min | PC skal være tændt | Åbner et link |
| **Stabilt link i dage/uger, gratis** | **B) Streamlit Cloud** + GitHub | ~20–40 min | Engang setup | Åbner `*.streamlit.app` |
| **Mere kontrol / længere tracks** | **C) Railway / Render + Docker** | ~1 time | Konto + kreditkort ofte | Samme: bare et link |

For en **ikke-teknisk veninde** er **B** det bedste “rigtige” valg.  
**A** er hurtigst hvis I sidder live.

---

## A) Hurtigst: del din lokale app med et link (tunnel)

Din computer kører AETHER. Et lille program laver et **offentligt https-link** til den.

### 1. Start AETHER som normalt

```powershell
cd "C:\Users\Morten\Documents\App concepts\Aether"
.\.venv\Scripts\Activate.ps1
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 2. Installer og kør en tunnel

**Mulighed 1 — Cloudflare Tunnel (anbefalet, gratis, ingen konto til hurtig test):**

```powershell
# engang: installer cloudflared
winget install Cloudflare.cloudflared

# i et NYT terminal-vindue (mens Streamlit kører):
cloudflared tunnel --url http://localhost:8501
```

Du får en URL ala `https://random-words.trycloudflare.com`  
→ den sender du til hende.

**Mulighed 2 — ngrok** (kræver gratis konto):

```powershell
winget install ngrok.ngrok
ngrok config add-authtoken <din-token-fra-dashboard>
ngrok http 8501
```

### Vigtigt for tunnel

- Din PC skal **være tændt** og appen køre  
- Linket dør når du lukker tunnel/PC  
- Del **kun** med folk du stoler på (alle med linket kan uploade audio til din maskine)  
- Brug korte tracks (1–3 min) så det ikke spiser al din CPU  

**Til hende:** “Åbn det her link i Chrome, upload en mp3, tryk Run analysis.”

---

## B) Stabilt gratis online: Streamlit Community Cloud (anbefalet)

Hun får et link som `https://dit-navn-aether.streamlit.app` der virker uden dig online.

### Forudsætninger

1. Gratis [GitHub](https://github.com)-konto  
2. Gratis [Streamlit Community Cloud](https://share.streamlit.io/) (login med GitHub)  
3. Dette projekt på GitHub (public repo er nemmest på free tier)

### Trin for dig

#### 1. Læg projektet på GitHub

Hvis du ikke har git repo endnu (i projektmappen):

```powershell
cd "C:\Users\Morten\Documents\App concepts\Aether"
git init
git add app.py requirements.txt packages.txt SETUP.md DEPLOY.md README.md .gitignore .streamlit .python-version Dockerfile aether
git commit -m "AETHER Analyzer ready for cloud"
```

Opret et **nyt public repo** på GitHub (fx `aether-analyzer`), **uden** README, og push:

```powershell
git branch -M main
git remote add origin https://github.com/<DIT-BRUGERNAVN>/aether-analyzer.git
git push -u origin main
```

> Upload **ikke** mappen `.venv` (den er i `.gitignore`).

#### 2. Deploy på Streamlit Cloud

1. Gå til https://share.streamlit.io/  
2. **New app**  
3. Vælg dit repo, branch `main`  
4. Main file: `app.py`  
5. Advanced → Python version **3.11** hvis muligt  
6. Deploy  

Streamlit installerer fra `requirements.txt` og systempakker fra `packages.txt` (ffmpeg m.m.).

#### 3. Send linket

Når status er **Running**, kopier URL’en til hende.

### Hvad hun skal gøre (copy-paste til hende)

```text
1. Åbn linket i Chrome eller Safari
2. I venstre side: upload en MP3 eller WAV (gerne under 3–4 minutter første gang)
3. Tryk "Run analysis" og vent (kan tage 30–120 sek)
4. Klik på events i listen — se DNA, type, preset-tekst
5. Prøv fanen "Resonance Map"
```

### Begrænsninger på free Streamlit Cloud

| Ting | Realitet |
|------|----------|
| RAM/CPU | Begrænset — lange tracks kan time out |
| Cold start | Første åbning kan tage 30–60 sek |
| Inaktivitet | App “sover”; vågner ved besøg |
| YouTube | Virker ofte (ffmpeg i packages.txt), men kan blokeres af host/net |
| Privacy | Public app = alle med link kan bruge den; upload er midlertidigt på deres servere |
| aubio/essentia | Essentia nej; aubio måske — librosa-fallback er fin |

**Tip:** Bed hende starte med et **kort** klip (30–90 sek) så første oplevelse er hurtig.

---

## C) Mere “rigtig” hosting: Docker på Railway / Render / Fly.io

Brug når Streamlit Cloud er for langsom, eller du vil have mere RAM.

Projektet har allerede en `Dockerfile`.

Eksempel **Railway**:

1. Push til GitHub (samme som B)  
2. New Project → Deploy from GitHub  
3. Port `8501`  
4. Public URL → send til hende  

Eksempel **lokalt test af Docker** (hvis du har Docker Desktop):

```powershell
docker build -t aether .
docker run -p 8501:8501 aether
```

---

## Privacy & “veninde-mode” tips

- Sig til hende: upload **ikke** private demos hun ikke vil have på en server  
- Tunnel (A) = data lander på **din** PC  
- Cloud (B/C) = midlertidigt på hostens maskine  
- Du kan senere tilføje et simpelt password i Streamlit (`st.secrets`) hvis du vil  

### Valgfri password (Streamlit Cloud)

I Streamlit Cloud → App settings → Secrets:

```toml
APP_PASSWORD = "jeres-hemmelige-kode"
```

(Password-gate skal kodes i `app.py` — sig til hvis du vil have det tilføjet.)

---

## Fejl du typisk ser

| Problem | Fix |
|---------|-----|
| Deploy fejler på dependencies | Python 3.11 + `requirements.txt` uden essentia/aubio som hard krav |
| App crasher på lang track | Kortere fil; hæv onset threshold i Advanced |
| “Please wait” for evigt | Cold start — genindlæs; free tier sover |
| YouTube virker ikke i cloud | Brug fil-upload i stedet |
| Hun kan ikke finde knappen | Send screenshot + de 5 trin ovenfor |

---

## Anbefaling til dig lige nu

1. **I aften / live demo:** metode **A (cloudflared)**  
2. **Så hun kan lege alene i weekenden:** metode **B (Streamlit Cloud)**  

Jeg kan også tilføje et lille **password-gate** og en venligere “Guest welcome”-banner i UI’en, hvis du vil gøre cloud-versionen endnu mere foolproof.
