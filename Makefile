.PHONY: generate-types test-contracts

generate-types:
	python tools/generate_ts_types.py

test-contracts:
	cd backend && python -m pytest tests/contracts/ -v
