.PHONY: figures test clean

figures:
	python run_analysis.py --output results

test:
	python -m unittest discover -s tests -v

clean:
	rm -rf results build dist *.egg-info
