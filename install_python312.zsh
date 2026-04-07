#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────
# Install Python 3.12.3 via pyenv — ZERO sudo required
# Everything lives inside ~/.pyenv
# ─────────────────────────────────────────────────────

set -e

# ── 1. Install pyenv ──────────────────────────────────
if [[ ! -d "$HOME/.pyenv" ]]; then
  echo "→ Installing pyenv..."
  curl -fsSL https://pyenv.run | zsh
else
  echo "→ pyenv already installed, skipping"
fi

# ── 2. Add pyenv to PATH (needed for this session) ───
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ── 3. Add pyenv to ~/.zshrc (permanent) ─────────────
ZSHRC="$HOME/.zshrc"

if ! grep -qF "PYENV_ROOT" "$ZSHRC" 2>/dev/null; then
  echo '' >> "$ZSHRC"
  echo '# pyenv' >> "$ZSHRC"
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$ZSHRC"
  echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$ZSHRC"
  echo 'eval "$(pyenv init -)"' >> "$ZSHRC"
  echo "  pyenv added to ~/.zshrc"
else
  echo "  pyenv already in ~/.zshrc, skipping"
fi

# ── 4. Install Python 3.12.3 ─────────────────────────
echo "→ Installing Python 3.12.3 (takes ~5–10 min)..."
pyenv install 3.12.3

# ── 5. Set as global default ──────────────────────────
echo "→ Setting Python 3.12.3 as global default..."
pyenv global 3.12.3

# ── Done ──────────────────────────────────────────────
echo ""
echo "✓ Done! Now run:"
echo "  source ~/.zshrc"
echo "  python --version   # → Python 3.12.3"