// static/js/script.js

// Page switching
function showHome() {
  document.getElementById('home').classList.add('active');
  document.getElementById('formPage').classList.remove('active');
}
function showForm() {
  document.getElementById('home').classList.remove('active');
  document.getElementById('formPage').classList.add('active');
}

document.addEventListener('DOMContentLoaded', () => {
  const goBtn = document.getElementById('go-to-form');
  const backBtn = document.getElementById('back-to-home');
  const form = document.getElementById('planForm');
  const btn = document.getElementById('generate-btn');
  const loading = document.getElementById('loading');
  const toast = document.getElementById('toast');

  if (goBtn) goBtn.addEventListener('click', showForm);
  if (backBtn) backBtn.addEventListener('click', showHome);

  // Basic UX for submission (server does full render)
  if (form) {
    form.addEventListener('submit', () => {
      if (btn) btn.disabled = true;
      if (loading) loading.classList.remove('hidden');
      // allow normal form submission; backend returns result page
    });
  }
});

// simple toast helper
function showToast(msg, type='success') {
  const t = document.getElementById('toast');
  if(!t) return;
  t.textContent = msg;
  t.className = 'toast show ' + type;
  setTimeout(()=> t.classList.remove('show'), 3000);
}

