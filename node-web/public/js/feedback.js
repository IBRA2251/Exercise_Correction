/**
 * Feedback System — visual toasts + voice via Web Speech API
 */

class FeedbackEngine {
    constructor() {
        this.cooldown         = 2.5;
        this.lastFeedbackTime = 0;
        this.lastMessage      = null;
        this.audioEnabled     = true;
        this.synth            = window.speechSynthesis || null;
        this.feedbackLog      = [];
        this._preferredVoice  = null;
        this._unlocked        = false;

        // Error tracking
        this.lastErrorKey          = null;
        this.consecutiveErrorCount = 0;
        this.ERROR_REPLAY_THRESHOLD= 2;

        this.onReplayTriggered = null;

        // Load voices — Chrome populates them asynchronously
        if (this.synth) {
            const load = () => {
                const voices = this.synth.getVoices();
                // prefer a local (offline) English voice for reliability
                this._preferredVoice =
                    voices.find(v => v.lang.startsWith('en') && v.localService) ||
                    voices.find(v => v.lang.startsWith('en')) ||
                    voices[0] || null;
            };
            load();
            this.synth.onvoiceschanged = load;
        }
    }

    // ─── Per-frame feedback ───────────────────────────────────────────────────
    generateFeedback(exerciseType, stage, angle, speed, modelStats) {
        if (!modelStats) return null;

        if (exerciseType === 'boxing') {
            if (stage === 'guard'   && angle < modelStats.max - 20)
                return { message:'Keep your guard up!',           errorKey:'guard_up'     };
            if (stage === 'punch'   && speed < 120)
                return { message:'Punch faster!',                 errorKey:'punch_faster' };
            if (stage === 'recover' && speed > 80)
                return { message:'Bring your hands back quicker', errorKey:'recover_slow' };
        }

        if (exerciseType === 'yoga') {
            if (stage === 'hold' && Math.abs(angle - modelStats.avg) > modelStats.std)
                return { message:'Hold steady, control your balance', errorKey:'hold_steady' };
        }

        if (exerciseType === 'single') {
            if (stage === 'bottom' && angle > modelStats.min + 15)
                return { message:'Go lower!',                        errorKey:'go_lower'  };
            if (stage === 'top'    && angle < modelStats.max - 15)
                return { message:'Fully extend at the top!',         errorKey:'extend_top' };
            if (speed > (modelStats.speed_avg || 200) + (modelStats.speed_std || 100) * 1.5)
                return { message:'Slow down — control the movement', errorKey:'slow_down' };
        }

        return null;
    }

    // ─── Error tracking ───────────────────────────────────────────────────────
    trackError(errorKey, message) {
        if (!errorKey) { this.lastErrorKey = null; this.consecutiveErrorCount = 0; return false; }
        if (errorKey === this.lastErrorKey) {
            this.consecutiveErrorCount++;
        } else {
            this.lastErrorKey = errorKey; this.consecutiveErrorCount = 1;
        }
        if (this.consecutiveErrorCount >= this.ERROR_REPLAY_THRESHOLD) {
            if (this.onReplayTriggered) this.onReplayTriggered(message);
            this.consecutiveErrorCount = 0; this.lastErrorKey = null;
            return true;
        }
        return false;
    }

    resetErrorTracking() { this.lastErrorKey = null; this.consecutiveErrorCount = 0; }

    // ─── UNLOCK — must be called synchronously inside a user-gesture handler ──
    // Speaks a real (non-empty) phrase so Chrome marks this page as speech-allowed.
    // After this runs, all subsequent synth.speak() calls work from any context.
    unlockAudio() {
        if (!this.synth) return;
        this.synth.cancel();
        const utt = new SpeechSynthesisUtterance('Session ready');
        utt.volume = 0.01;   // barely audible — just enough to not be rejected
        utt.rate   = 2.5;    // very fast so the user doesn't hear it
        if (this._preferredVoice) utt.voice = this._preferredVoice;
        utt.onerror = () => {};
        this.synth.speak(utt);
        this._unlocked = true;
    }

    // ─── DELIVER — audio + visual ─────────────────────────────────────────────
    deliver(message, isGoodFeedback = false) {
        if (!message) return false;
        const now = performance.now() / 1000;
        if (message === this.lastMessage && now - this.lastFeedbackTime < this.cooldown) return false;

        this.lastMessage      = message;
        this.lastFeedbackTime = now;

        // Log
        const ts = new Date().toLocaleTimeString('en-US',
            { hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
        this.feedbackLog.unshift({ time:ts, text:message, good:isGoodFeedback });
        if (this.feedbackLog.length > 50) this.feedbackLog.pop();

        // Audio
        if (this.audioEnabled && this.synth) {
            this._speak(message);
        }

        this._showToast(message, isGoodFeedback);
        this._updateLogUI();
        return true;
    }

    _speak(text) {
        if (!this.synth) return;
        // Chrome: if the engine paused (tab lost focus), resume it first
        if (this.synth.paused) this.synth.resume();
        // Clear any queued or current speech so new message starts immediately
        if (this.synth.speaking || this.synth.pending) this.synth.cancel();

        const utt = new SpeechSynthesisUtterance(text);
        utt.rate  = 1.1;
        utt.pitch = 1.0;
        utt.volume = 1.0;
        if (this._preferredVoice) utt.voice = this._preferredVoice;
        utt.onerror = () => {};
        this.synth.speak(utt);
    }

    // ─── TOGGLE — returns true when audio is ON ───────────────────────────────
    toggleAudio() {
        this.audioEnabled = !this.audioEnabled;
        if (!this.audioEnabled && this.synth) {
            this.synth.cancel();
        }
        return this.audioEnabled;
    }

    // ─── TOAST ────────────────────────────────────────────────────────────────
    _showToast(message, isGood) {
        const toast = document.getElementById('feedback-toast');
        if (!toast) return;
        toast.textContent = message;
        toast.className   = 'feedback-toast' + (isGood ? ' good' : '');
        toast.classList.add('visible');
        clearTimeout(this._toastTimeout);
        this._toastTimeout = setTimeout(() => toast.classList.remove('visible'), 2500);
    }

    // ─── FEEDBACK LOG UI ─────────────────────────────────────────────────────
    _updateLogUI() {
        const container = document.getElementById('feedback-log');
        if (!container) return;
        if (!this.feedbackLog.length) {
            container.innerHTML = '<div class="feedback-empty">Feedback appears here during your session</div>';
            return;
        }
        container.innerHTML = this.feedbackLog.slice(0, 12).map(f => {
            const pos = f.good ||
                ['good','great','perfect','excellent','clean','solid','sharp','beautiful']
                    .some(w => f.text.toLowerCase().includes(w));
            const cls = pos ? 'fb-good' : 'fb-error';
            return `<div class="feedback-item ${cls}">
                <span class="fb-time">${f.time}</span>
                <span class="fb-text">${f.text}</span>
            </div>`;
        }).join('');
    }

    reset() {
        this.feedbackLog = []; this.lastMessage = null; this.lastFeedbackTime = 0;
        this.lastErrorKey = null; this.consecutiveErrorCount = 0;
        if (this.synth && (this.synth.speaking || this.synth.pending)) this.synth.cancel();
        this._updateLogUI();
    }
}

// ─── Rep celebration ─────────────────────────────────────────────────────────
function celebrateRep(count) {
    const el = document.getElementById('rep-count');
    if (el) { el.classList.remove('rep-pop'); void el.offsetWidth; el.classList.add('rep-pop'); }
    const vc = document.getElementById('video-container');
    if (vc) { vc.style.boxShadow = '0 0 50px rgba(16,185,129,0.45)'; setTimeout(() => { vc.style.boxShadow = ''; }, 500); }
}

const _repPopStyle = document.createElement('style');
_repPopStyle.textContent = `
@keyframes repPop {
    0%   { transform:translate(-50%,-50%) scale(1); }
    40%  { transform:translate(-50%,-50%) scale(1.4); }
    70%  { transform:translate(-50%,-50%) scale(.95); }
    100% { transform:translate(-50%,-50%) scale(1); }
}
.rep-pop { animation:repPop .5s cubic-bezier(.34,1.56,.64,1); }
`;
document.head.appendChild(_repPopStyle);

window.FeedbackEngine = FeedbackEngine;
window.celebrateRep   = celebrateRep;
