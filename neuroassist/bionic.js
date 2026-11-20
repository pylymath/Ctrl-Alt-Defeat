// Bionic reading for the WHOLE page: labels, buttons, chips, tabs, settings, headings and the example
// text inside empty boxes. Skipped: the "Percie" title, code, and the answer areas (.na-out), which the
// server already renders in bionic form. Text you type into a box cannot be partly bold (a browser limit).
// window.percieBionic(true/false) is called by the "Bionic reading" checkbox.
(() => {
  const SKIP = "script,style,textarea,input,select,option,svg,code,pre,.na-code,.na-head h1,.na-out,.pb,.pb-ph," +
               "[contenteditable],.na-cl";
  let on = false;

  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // Same rule as the server: bold the first half (rounded up) of every word made of letters.
  const bionic = (text) => esc(text).replace(/[A-Za-zÀ-ɏ]+/g, (w) => {
    const k = Math.ceil(w.length / 2);
    return "<b>" + w.slice(0, k) + "</b>" + w.slice(k);
  });

  function wrapText(root) {
    if (root && root.nodeType === Node.TEXT_NODE) root = root.parentNode;
    if (!root || root.nodeType !== Node.ELEMENT_NODE) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue || !/[A-Za-z]/.test(n.nodeValue)) return NodeFilter.FILTER_REJECT;
        const p = n.parentElement;
        if (!p || p.closest(SKIP)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const n of nodes) {
      const s = document.createElement("span");
      s.className = "pb";
      s.dataset.o = n.nodeValue;
      s.innerHTML = bionic(n.nodeValue);
      n.parentNode.replaceChild(s, n);
    }
  }

  function unwrapText() {
    document.querySelectorAll("span.pb").forEach((s) => s.replaceWith(document.createTextNode(s.dataset.o)));
  }

  // Placeholders cannot hold bold text, so while bionic is on an overlay shows them instead.
  function placeholdersOn() {
    document.querySelectorAll("textarea[placeholder], input[placeholder], textarea[data-pb-ph], input[data-pb-ph]")
      .forEach((el) => {
        const ph = el.dataset.pbPh ?? el.getAttribute("placeholder");
        if (!ph || el.type === "number" || el.offsetParent === null) return;
        el.dataset.pbPh = ph;
        el.setAttribute("placeholder", "");
        let o = el.nextElementSibling;
        if (!o || !o.classList.contains("pb-ph")) {
          o = document.createElement("div");
          o.className = "pb-ph";
          el.after(o);
          if (getComputedStyle(el.parentElement).position === "static") el.parentElement.style.position = "relative";
        }
        const cs = getComputedStyle(el);
        o.style.cssText = `position:absolute;pointer-events:none;left:${el.offsetLeft}px;top:${el.offsetTop}px;` +
          `width:${el.clientWidth}px;max-height:${el.clientHeight}px;padding:${cs.padding};font-family:${cs.fontFamily};` +
          `font-size:${cs.fontSize};line-height:${cs.lineHeight};color:${cs.color};opacity:.55;white-space:pre-wrap;` +
          `overflow:hidden;box-sizing:border-box;`;
        if (o.dataset.text !== ph) {           // only rewrite when it changed: rewriting is itself a DOM change
          o.dataset.text = ph;
          o.innerHTML = bionic(ph);
        }
        o.style.display = el.value ? "none" : "block";
      });
  }

  function placeholdersOff() {
    document.querySelectorAll(".pb-ph").forEach((o) => o.remove());
    document.querySelectorAll("[data-pb-ph]").forEach((el) => {
      el.setAttribute("placeholder", el.dataset.pbPh);
      delete el.dataset.pbPh;
    });
  }

  // New text (opened sections, updated step lists, new answers' surroundings) gets the same treatment.
  // Our own changes (wrapping text, overlays) are ignored, so this can never feed itself.
  const ours = (n) => {
    const el = n.nodeType === Node.ELEMENT_NODE ? n : n.parentElement;
    return !!(el && el.closest && el.closest(".pb, .pb-ph"));
  };
  let pending = false;
  const observer = new MutationObserver((muts) => {
    if (!on) return;
    const fresh = [];
    for (const m of muts) {
      if (ours(m.target)) continue;
      m.addedNodes.forEach((n) => { if (!ours(n)) fresh.push(n); });
    }
    if (!fresh.length || pending) return;
    pending = true;
    setTimeout(() => {                     // batch: at most one pass per burst of changes
      pending = false;
      if (!on) return;
      observer.disconnect();
      fresh.forEach(wrapText);
      placeholdersOn();
      observer.observe(document.body, { childList: true, subtree: true });
    }, 30);
  });

  // Hide an overlay as soon as its box has text (typed or filled in by a chip).
  setInterval(() => {
    if (!on) return;
    document.querySelectorAll(".pb-ph").forEach((o) => {
      const el = o.previousElementSibling;
      if (el) o.style.display = el.value ? "none" : "block";
    });
  }, 250);

  // Opening a section or switching tabs only shows hidden boxes (no new nodes): refresh after clicks.
  document.addEventListener("click", () => {
    if (!on) return;
    setTimeout(() => {
      if (!on) return;                       // it may have been switched off in the meantime
      observer.disconnect();
      wrapText(document.body);
      placeholdersOn();
      observer.observe(document.body, { childList: true, subtree: true });
    }, 120);
  }, true);

  window.percieBionic = (value) => {
    on = !!value;
    document.body.classList.toggle("percie-bionic", on);
    if (on) {
      observer.disconnect();
      wrapText(document.body);
      placeholdersOn();
      observer.observe(document.body, { childList: true, subtree: true });
    } else {
      observer.disconnect();
      unwrapText();
      placeholdersOff();
    }
  };
})();
