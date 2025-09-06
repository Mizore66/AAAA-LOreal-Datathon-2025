#!/bin/bash
# Example external agent script for automated pipeline execution
# This demonstrates how an external system can delegate pipeline execution

echo "=== EXTERNAL AGENT PIPELINE EXECUTION ==="
echo "Timestamp: $(date)"
echo "Agent: External Automation System"
echo

# Change to project directory
cd "$(dirname "$0")"

# Execute pipeline with custom message
python pipeline_executor.py --commit-message "Automated execution by external agent - $(date '+%Y-%m-%d %H:%M:%S')"

# Check exit status
if [ $? -eq 0 ]; then
    echo
    echo "✅ Pipeline execution completed successfully"
    echo "📊 Generated files summary available in execution_summary.json"
    echo "📝 Detailed logs available in pipeline_execution.log"
    
    # Optional: Send notification or trigger downstream processes
    echo "🔔 External agent could now send notifications or trigger downstream processes"
    
else
    echo
    echo "❌ Pipeline execution failed"
    echo "📋 Check pipeline_execution.log for detailed error information"
    echo "🚨 External agent should handle error notifications and recovery"
fi

echo
echo "=== EXTERNAL AGENT EXECUTION COMPLETE ==="