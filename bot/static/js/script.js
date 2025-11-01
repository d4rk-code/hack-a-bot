// bot/static/js/script.js
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("plan-form");
  const btn = document.getElementById("generate-btn");
  const loading = document.getElementById("loading");

  if (!form) return;

  form.addEventListener("submit", (e) => {
    // If you want classic form submit (server render), skip AJAX and let Django handle
    // This JS only shows UX improvements while form submits
    btn.disabled = true;
    loading.classList.remove("hidden");
  });
});

