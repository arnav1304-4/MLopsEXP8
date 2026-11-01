"""Container entry point: start the Prometheus endpoint, then Streamlit in the same process.

Starting the metrics server here (instead of inside app.py) means /metrics is
available as soon as the container starts, before any browser session opens.
"""
import sys

from streamlit.web import cli as stcli

import monitoring

if __name__ == "__main__":
    monitoring.load_saved_model_metrics()
    monitoring.start_metrics_server()
    sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
    sys.exit(stcli.main())
