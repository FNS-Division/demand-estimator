<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="15%">
</p>
<p align="center"><h1 align="center">DEMAND-ESTIMATOR</h1></p>
<p align="center">
	<em><code>A tool for estimatating internet connectivity demand for points of interest such as schools or hospitals.</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/FNS-Division/demand-estimator?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/FNS-Division/demand-estimator?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/FNS-Division/demand-estimator?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/FNS-Division/demand-estimator?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>❯ This tool processes point of interest (POI) data to analyze internet connectivity demand, expressed as throughput in megabits per second. It calculates the potential number of users from population density data, applying configurable parameters such as demand per user in mbps, radius settings and population data source.</code>

---

##  Project Structure

```sh
└── demand-estimator/
    ├── LICENSE
    ├── README.md
    ├── data
    │   └── ESP
    ├── demand
    │   ├── dataprocess.py
    │   ├── demand
    │   ├── entities
    │   ├── handlers
    │   └── utils.py
    └── notebooks
        └── run_demand.ipynb
```


###  Project Index
<details open>
	<summary><b><code>DEMAND-ESTIMATOR/</code></b></summary>
	<details> <!-- notebooks Submodule -->
		<summary><b>notebooks</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/FNS-Division/demand-estimator/blob/master/notebooks/run_demand.ipynb'>run_demand.ipynb</a></b></td>
				<td><code>❯ Notebook to demonstrate how to use the tool</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- demand Submodule -->
		<summary><b>demand</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/FNS-Division/demand-estimator/blob/master/demand/utils.py'>utils.py</a></b></td>
				<td><code>❯ Helper functions</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/FNS-Division/demand-estimator/blob/master/demand/dataprocess.py'>dataprocess.py</a></b></td>
				<td><code>❯ Data processing functions</code></td>
			</tr>
			</table>
			<details>
				<summary><b>handlers</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/FNS-Division/demand-estimator/blob/master/demand/handlers/populationdatahandler.py'>populationdatahandler.py</a></b></td>
						<td><code>❯ Handler to source population data from WorldPop</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>entities</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/FNS-Division/demand-estimator/blob/master/demand/entities/pointofinterest.py'>pointofinterest.py</a></b></td>
						<td><code>❯ Classes to create point of interest collections</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/FNS-Division/demand-estimator/blob/master/demand/entities/entity.py'>entity.py</a></b></td>
						<td><code>❯ Classes to create generic collections</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>demand</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/FNS-Division/demand-estimator/blob/master/demand/demand/demand.py'>demand.py</a></b></td>
						<td><code>❯ Main module for population and demand estimation</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with demand-estimator, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


###  Installation

Install demand-estimator using one of the following methods:

**Build from source:**

1. Clone the demand-estimator repository:
```sh
❯ git clone https://github.com/FNS-Division/demand-estimator
```

2. Navigate to the project directory:
```sh
❯ cd demand-estimator
```

3. Install the project dependencies and set up the conda environment:

```sh
❯ conda env create -f environment.yml
❯ conda activate demand-env
```



###  Usage

The tool is meant to be used by launching the provided [notebook](notebooks\run_demand.ipynb), which serves as a template.

The following parameters need to be configured:

| Parameter | Description |
|-----------|-------------|
| `country_code` | Three-letter ISO3 country code (e.g., 'ESP' for Spain) used to identify the geographical region for analysis |
| `poi_dataset_id` | Filename of the Point of Interest dataset containing locations to analyze |
| `radii` | List of distances (in km) used to create buffer zones around each POI for analysis |
| `radius_for_demand` | Specific radius (in km) used for calculating demand metrics |
| `dataset_year` | Year of the population dataset to be used in the analysis |
| `one_km_res` | Boolean flag indicating whether to use 1km resolution population data (True) or lower resolution (False) |
| `un_adjusted` | Boolean flag for using UN-adjusted population data (True) or unadjusted data (False) |
| `overlap_allowed` | Boolean flag determining if POI buffer zones can overlap in the analysis (False = no overlap) |
| `mbps_demand_per_user` | The bandwidth demand in Mbps assigned to each user in the calculation |
| `are_poi_schools` | Boolean flag indicating whether the POIs represent schools (affects demand calculations) |

If `are_poi_schools` is set to `True`, then the tool will:

- Fetch the population of compulsory school age in that country and year (Source: [UNESCO](https://data.uis.unesco.org/index.aspx?queryid=3847)).
- Disaggregatate this number across all the schools in the provided dataset, based on how many people live in the catchment area around each school.
- For example, if 2% of Spain's population lives in a 1km radius around a school, then 2% of Spain's population of compulsory school age is assumed to be a user of this school.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/FNS-Division/demand-estimator/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/FNS-Division/demand-estimator/issues)**: Submit bugs found or log feature requests for the `demand-estimator` project.
- **💡 [Submit Pull Requests](https://github.com/FNS-Division/demand-estimator/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/FNS-Division/demand-estimator
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/FNS-Division/demand-estimator/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=FNS-Division/demand-estimator">
   </a>
</p>
</details>

---

##  License

This project is protected under the [MIT](LICENSE) License. For more details, refer to the [LICENSE](LICENSE) file.

---
