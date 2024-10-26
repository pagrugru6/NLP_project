.PHONY: isort black ruff ruff_fix clean lstm lstm_ out cancel queue load

format: ## Execute pre-commit hooks to enforce code style.
	@echo "Running pre-commit hooks for code formatting..."
	@pre-commit run -a
	@echo "Code formatting completed."

clean:
	rm -f ../slurm_out/*.out

lstm:
	@sbatch sbatchbilstm.sh

# If we dont want to wait for the queue and we don't need any major ressources
_lstm:
	python run_bilstm.py

load:
	@echo "module load anaconda3/2024.06-py3.12.4"
	@echo "module load cuda/11.8"
	@echo "conda activate nlp"

# @ hides the command when we use it in the terminal
# $$ is necessary when we want to use multiple commands and concat them.
# In a Makefile each line of a recipe is executed in a separated shell instance. so you need to escape the '$' sign to prevent make from interpreting it as a makefile varialbe. Use '$$' to rpresent a single '$'
out:
	@echo "################## You are looking at file: $$(ls -rt ../slurm_out/ | tail -n1) ##################" ""
	@cat ../slurm_out/$$(ls -rt ../slurm_out/ | tail -n1)
	@echo "You were looking at file: $$(ls -rt ../slurm_out/ | tail -n1) "

cancel:
	scancel -u hlp179

queue:
	squeue -u hlp179
