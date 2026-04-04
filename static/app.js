const input = document.getElementById('query');
const chips = document.querySelectorAll('.chip');
const form = document.getElementById('ask-form');
const copyBtn = document.querySelector('.copy-answer');
const answerBlock = document.getElementById('answer-block');
const answerPanel = document.getElementById('answer-panel');

chips.forEach((chip) => {
  chip.addEventListener('click', () => {
    input.value = chip.dataset.q || '';
    input.focus();
  });
});

if (form) {
  form.addEventListener('submit', () => {
    const btn = form.querySelector('button[type="submit"]');
    if (btn) {
      btn.textContent = 'جاري التحليل...';
      btn.disabled = true;
    }
  });
}

if (copyBtn && answerBlock) {
  copyBtn.addEventListener('click', async () => {
    const text = (answerBlock.innerText || '').trim();
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      const old = copyBtn.textContent;
      copyBtn.textContent = 'تم النسخ';
      setTimeout(() => {
        copyBtn.textContent = old;
      }, 1200);
    } catch (_) {
      copyBtn.textContent = 'انسخ يدويًا';
    }
  });
}

if (answerPanel && answerBlock && (answerBlock.innerText || '').trim()) {
  window.requestAnimationFrame(() => {
    answerPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    answerPanel.classList.add('reveal');
  });
}
