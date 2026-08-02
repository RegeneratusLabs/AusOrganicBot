#!/bin/bash
# AusOrganicBot — One-Click Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configurable port (override via env, e.g. PORT=8000 ./start.sh)
PORT="${PORT:-8080}"

echo "============================================================"
echo "🌱 AusOrganicBot — Starting..."
echo "============================================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "  ✗ Python 3 not found"; exit 1
fi
echo "  ✓ Python 3 found ($(python3 --version))"

if ! command -v node &> /dev/null; then
    echo "  ✗ Node.js not found"; exit 1
fi
echo "  ✓ Node.js found ($(node --version))"

# Check API key
if [ -f "backend/.env" ]; then
    if grep -q "LLM_API_KEY=" backend/.env; then
        echo "  ✓ LLM API key configured"
    else
        echo "  ⚠ LLM API key not set in backend/.env"
        echo "    Edit backend/.env and add your LLM_API_KEY"
    fi
else
    echo "  ⚠ No backend/.env file — run: cp backend/.env.example backend/.env"
fi

echo ""

# Kill any existing processes
pkill -f "uvicorn app.main:app" 2>/dev/null
pkill -f "next dev" 2>/dev/null
sleep 1

# Install backend deps
echo "Checking backend dependencies..."
cd "$SCRIPT_DIR/backend"
pip3 install -r requirements.txt --quiet 2>/dev/null
echo "  ✓ Backend dependencies ready"

# Install frontend deps
echo "Checking frontend dependencies..."
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    npm install --silent
    echo "  ✓ Frontend dependencies installed"
else
    echo "  ✓ Frontend dependencies ready"
fi

echo ""

# Start backend
echo "[1/2] Starting backend on http://localhost:${PORT}..."
cd "$SCRIPT_DIR/backend"
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port "$PORT" > /tmp/ausorganicbot-backend.log 2>&1 &
BACKEND_PID=$!

echo "  Waiting for backend..."
for i in {1..30}; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  ✓ Backend ready"
        break
    fi
    sleep 2
done

if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "  ✗ Backend failed to start"
    echo "  Check logs: /tmp/ausorganicbot-backend.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo "[2/2] Starting frontend on http://localhost:3000..."
cd "$SCRIPT_DIR/frontend"
npm run dev > /tmp/ausorganicbot-frontend.log 2>&1 &
FRONTEND_PID=$!

sleep 5

echo ""
echo "============================================================"
echo "✅ AusOrganicBot is running!"
echo "============================================================"
echo ""
echo "  🌐 App:       http://localhost:3000"
echo "  🔧 Backend:   http://localhost:${PORT}"
echo "  📚 API Docs:  http://localhost:${PORT}/docs"
echo ""
echo "  Backend PID:  $BACKEND_PID"
echo "  Frontend PID: $FRONTEND_PID"
echo ""
echo "  Logs: /tmp/ausorganicbot-backend.log, /tmp/ausorganicbot-frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo "============================================================"

xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null || true

trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '✓ Stopped'; exit" INT

wait
