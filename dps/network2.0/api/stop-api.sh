
#!/bin/bash
# Stop the API server

echo "Stopping Network Monitoring API server..."
pkill -f "python.*api.py"
echo "API server stopped"