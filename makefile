activate-venv:
	@echo "Activating virtual environment..."
	source venv/bin/activate
	
run-app:
	@echo "Running the application..."
	chainlit run app.py --watch
