# SPDX-FileCopyrightText: 2024 ZB MED - Information Centre for Life Sciences
# SPDX-FileCopyrightText: 2024 Benjamin Wolff
#
# SPDX-License-Identifier: CC-BY-4.0

formatting:
	@python -m black notebooks --line-length=120

check-formatting:
	@python -m black notebooks --check --line-length=120

diff-formatting:
	@python -m black notebooks --diff --line-length=120

check-code:
	@python -m flake8 notebooks

audit: check-formatting check-code