{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMHk67/11flaAcLOSP6lpBI",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/selmirayen870-wq/titanic_survival_prediction/blob/main/titanic_survival_prediction.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XTwi6clC7vNx",
        "outputId": "2a3d87da-478c-4ec0-bdf9-343903d5ae3e",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2.2.3 2.1.3\n"
          ]
        }
      ],
      "source": [
        "   import pandas as pd\n",
        "   import numpy as np\n",
        "   print(pd.__version__, np.__version__)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "url = \"https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv\"\n",
        "df = pd.read_csv(url)\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "roTNCqAP--h5",
        "outputId": "bf41bb5e-2e02-41ed-aa52-2a918a36d114"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-6246ffb6-7c9d-431e-9dc7-77abb001091a\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-6246ffb6-7c9d-431e-9dc7-77abb001091a')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-6246ffb6-7c9d-431e-9dc7-77abb001091a button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-6246ffb6-7c9d-431e-9dc7-77abb001091a');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 891,\n  \"fields\": [\n    {\n      \"column\": \"PassengerId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 257,\n        \"min\": 1,\n        \"max\": 891,\n        \"num_unique_values\": 891,\n        \"samples\": [\n          710,\n          440,\n          841\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Survived\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          3,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Name\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 891,\n        \"samples\": [\n          \"Moubarek, Master. Halim Gonios (\\\"William George\\\")\",\n          \"Kvillner, Mr. Johan Henrik Johannesson\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"female\",\n          \"male\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 14.526497332334044,\n        \"min\": 0.42,\n        \"max\": 80.0,\n        \"num_unique_values\": 88,\n        \"samples\": [\n          0.75,\n          22.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 8,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Ticket\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 681,\n        \"samples\": [\n          \"11774\",\n          \"248740\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 49.693428597180905,\n        \"min\": 0.0,\n        \"max\": 512.3292,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          11.2417,\n          51.8625\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Cabin\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 147,\n        \"samples\": [\n          \"D45\",\n          \"B49\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"S\",\n          \"C\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XeYJVhKJ_LF8",
        "outputId": "4c330193-acab-4c54-a16c-1617c10c6d53"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(891, 12)"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XsRpTV85_OhN",
        "outputId": "bb9f18d7-10e7-466f-8181-99412533c74e"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 12 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  891 non-null    int64  \n",
            " 1   Survived     891 non-null    int64  \n",
            " 2   Pclass       891 non-null    int64  \n",
            " 3   Name         891 non-null    object \n",
            " 4   Sex          891 non-null    object \n",
            " 5   Age          714 non-null    float64\n",
            " 6   SibSp        891 non-null    int64  \n",
            " 7   Parch        891 non-null    int64  \n",
            " 8   Ticket       891 non-null    object \n",
            " 9   Fare         891 non-null    float64\n",
            " 10  Cabin        204 non-null    object \n",
            " 11  Embarked     889 non-null    object \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 83.7+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isna().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 460
        },
        "id": "oMJznnM2_SPC",
        "outputId": "7f1d62a8-d5e8-4409-ae90-b97f8440a9f6"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PassengerId      0\n",
              "Survived         0\n",
              "Pclass           0\n",
              "Name             0\n",
              "Sex              0\n",
              "Age            177\n",
              "SibSp            0\n",
              "Parch            0\n",
              "Ticket           0\n",
              "Fare             0\n",
              "Cabin          687\n",
              "Embarked         2\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>PassengerId</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Survived</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Pclass</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Name</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sex</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Age</th>\n",
              "      <td>177</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>SibSp</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Parch</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Ticket</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Fare</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Cabin</th>\n",
              "      <td>687</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Embarked</th>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby(\"Sex\")[\"Survived\"].mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "zd7AVfkq_T8N",
        "outputId": "d696a51c-be85-4e3c-95c8-33a33875b234"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Sex\n",
              "female    0.742038\n",
              "male      0.188908\n",
              "Name: Survived, dtype: float64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sex</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>female</th>\n",
              "      <td>0.742038</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>male</th>\n",
              "      <td>0.188908</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby(\"Pclass\")[\"Survived\"].mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 210
        },
        "id": "X9--5PdM_atZ",
        "outputId": "8bf9b4c8-f35a-4332-c75b-174da1e15f57"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Pclass\n",
              "1    0.629630\n",
              "2    0.472826\n",
              "3    0.242363\n",
              "Name: Survived, dtype: float64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Pclass</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.629630</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.472826</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.242363</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Embarked : remplacer par le mode\n",
        "df[\"Embarked\"] = df[\"Embarked\"].fillna(df[\"Embarked\"].mode()[0])"
      ],
      "metadata": {
        "id": "xm6jpAQEAJC8"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 2. Age : médiane par groupe (sexe + classe)\n",
        "df[\"Age\"] = df[\"Age\"].fillna(\n",
        "    df.groupby([\"Sex\", \"Pclass\"])[\"Age\"].transform(\"median\")\n",
        ")"
      ],
      "metadata": {
        "id": "_GqP_cVtAOzP"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 3. Cabin : créer une variable indiquant si la cabine est connue, puis supprimer l'originale\n",
        "df[\"HasCabin\"] = df[\"Cabin\"].notna().astype(int)\n",
        "df = df.drop(columns=[\"Cabin\"])"
      ],
      "metadata": {
        "id": "cXTWFj_yAR2-"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 4. Vérification : tout doit être à 0\n",
        "df.isna().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 460
        },
        "id": "GwERZBncAU_0",
        "outputId": "bf41fd45-2c47-40dd-e002-f8c92b5551a7"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PassengerId    0\n",
              "Survived       0\n",
              "Pclass         0\n",
              "Name           0\n",
              "Sex            0\n",
              "Age            0\n",
              "SibSp          0\n",
              "Parch          0\n",
              "Ticket         0\n",
              "Fare           0\n",
              "Embarked       0\n",
              "HasCabin       0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>PassengerId</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Survived</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Pclass</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Name</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sex</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Age</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>SibSp</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Parch</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Ticket</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Fare</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Embarked</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>HasCabin</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby(\"HasCabin\")[\"Survived\"].mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "RoRUnKmoAZJg",
        "outputId": "f88e778c-6a52-4178-ff97-d43a4177a8e0"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "HasCabin\n",
              "0    0.299854\n",
              "1    0.666667\n",
              "Name: Survived, dtype: float64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>HasCabin</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.299854</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.666667</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby([\"Pclass\", \"HasCabin\"])[\"Survived\"].agg([\"mean\", \"count\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "GI8U5JbTAyLb",
        "outputId": "24c32dbe-0c28-4a7c-8bab-722955e52a56"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                     mean  count\n",
              "Pclass HasCabin                 \n",
              "1      0         0.475000     40\n",
              "       1         0.664773    176\n",
              "2      0         0.440476    168\n",
              "       1         0.812500     16\n",
              "3      0         0.235908    479\n",
              "       1         0.500000     12"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8b696243-d0c5-4644-bf60-9b42cb06249c\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th>mean</th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Pclass</th>\n",
              "      <th>HasCabin</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th rowspan=\"2\" valign=\"top\">1</th>\n",
              "      <th>0</th>\n",
              "      <td>0.475000</td>\n",
              "      <td>40</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.664773</td>\n",
              "      <td>176</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th rowspan=\"2\" valign=\"top\">2</th>\n",
              "      <th>0</th>\n",
              "      <td>0.440476</td>\n",
              "      <td>168</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.812500</td>\n",
              "      <td>16</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th rowspan=\"2\" valign=\"top\">3</th>\n",
              "      <th>0</th>\n",
              "      <td>0.235908</td>\n",
              "      <td>479</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.500000</td>\n",
              "      <td>12</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8b696243-d0c5-4644-bf60-9b42cb06249c')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-8b696243-d0c5-4644-bf60-9b42cb06249c button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-8b696243-d0c5-4644-bf60-9b42cb06249c');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 6,\n  \"fields\": [\n    {\n      \"column\": \"mean\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.19796975620833193,\n        \"min\": 0.2359081419624217,\n        \"max\": 0.8125,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.475,\n          0.6647727272727273,\n          0.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 177,\n        \"min\": 12,\n        \"max\": 479,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          40,\n          176,\n          12\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
        "\n",
        "sns.barplot(data=df, x=\"Sex\", y=\"Survived\", ax=axes[0])\n",
        "axes[0].set_title(\"Survie par sexe\")\n",
        "\n",
        "sns.barplot(data=df, x=\"Pclass\", y=\"Survived\", ax=axes[1])\n",
        "axes[1].set_title(\"Survie par classe\")\n",
        "\n",
        "sns.barplot(data=df, x=\"HasCabin\", y=\"Survived\", ax=axes[2])\n",
        "axes[2].set_title(\"Survie selon HasCabin\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 323
        },
        "id": "yQwJ9kF7A2fr",
        "outputId": "9724c4ea-774a-4881-a160-34b39353aa22"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1500x400 with 3 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABdIAAAGGCAYAAAB/pnNVAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAbgJJREFUeJzt3XtYVNX+x/HPgAIiFzUEjEhSUyMTDISoTE9hVObRrmgXkZRzfiqdasqKLpBaYmmGxyzKJO0qWZYdNbQoKpPC8FhZXtJSrAQkFZRqSNi/P3qc0wSMgAPD5f16nv087rXX2vu7J+jLfGfPWibDMAwBAAAAAAAAAIA6uTg7AAAAAAAAAAAAWjMK6QAAAAAAAAAA2EEhHQAAAAAAAAAAOyikAwAAAAAAAABgB4V0AAAAAAAAAADsoJAOAAAAAAAAAIAdFNIBAAAAAAAAALCDQjoAAAAAAAAAAHZQSAcAAAAAAAAAwA4K6UAHMHHiRIWEhDg7DAAA8BcdNUd31PsGALSc1pxrHnroIZlMJmeH0S7s2bNHJpNJ8+bNO2FfXnecLArpgIN99dVXuvbaa9W7d295eHgoKChII0eO1MKFC50dGgAAHRo5GgCA5kGObTtMJpOSk5PrPLZ06VKZTCZ9/vnnzRpDXl6err76agUGBsrNzU3+/v4aPXq0Vq5c2azXBU4WhXTAgTZu3KjIyEh98cUXSkpK0pNPPqnJkyfLxcVFCxYscFpcixcv1o4dO5x2fQAAnI0cDQBA8yDHojHS0tL0t7/9TVu3btU///lPZWZmavr06Tp69KiuueYavfLKK8127QceeEC//vprs50f7V8nZwcAtCePPPKIfH19tWnTJnXr1s3mWGlpqcOuU1lZqa5duza4f+fOnR127ZZSU1OjqqoqeXh4ODsUAEA7QI52HHI0AODPyLFoqNdff10zZ87Utddeq1deecXmv9H06dO1bt06/f777812/U6dOqlTJ0qhaDqeSAccaPfu3Tr77LNr/fEgSf7+/tZ/H5/Da+nSpbX6mUwmPfTQQ9b943N4ffPNN7rhhhvUvXt3XXjhhZo3b55MJpP27t1b6xwpKSlyc3PToUOHJNU9N1xNTY0yMjJ09tlny8PDQwEBAfrnP/9pHWPPxIkT5eXlpe+++05xcXHq2rWrTj31VM2cOVOGYdj0nTdvns4//3ydcsop6tKliyIiIvT666/Xed/Jycl6+eWXdfbZZ8vd3V05OTn1xvD5558rLi5Ofn5+6tKli8444wzdcsstjb7HtLQ0ubi4KDc312bsP/7xD7m5uemLL76wtn322We67LLL5OvrK09PTw0fPlyffPLJCV8vAIDzkaNbLkdL0jvvvKPhw4fL29tbPj4+Gjp06AmfMGtoPO+++64uvPBCdevWTV5eXhowYIDuu+8+mz4LFy7U2WefLU9PT3Xv3l2RkZG1rv/jjz/qlltuUUBAgNzd3XX22WcrKyvLbowAgNo6So4tLi5WYmKiTjvtNLm7u6tXr14aM2aM9uzZY9PvnXfe0bBhw9S1a1d5e3tr1KhR+vrrr094/mPHjmnWrFnq27ev3N3dFRISovvuu08Wi8WmX0hIiK688kpt2LBBUVFR8vDwUJ8+ffTCCy+c8BpN8eWXX2rixInq06ePPDw8FBgYqFtuuUU///yzTb8jR47o9ttvV0hIiNzd3eXv76+RI0dq8+bN1j4PPvigevTooaysrDo/6IiLi9OVV14pSaqqqlJqaqoiIiLk6+urrl27atiwYfrggw/qjfWJJ55Q79691aVLFw0fPlxbt261OV7XHOnH/8556623NGjQIOvfBCf6WwcdE4V0wIF69+6twsLCWv+zdoTrrrtOv/zyi2bPnq2kpCRdf/31MplMeu2112r1fe2113TppZeqe/fu9Z7vn//8p6ZPn64LLrhACxYsUGJiol5++WXFxcU16BPg6upqXXbZZQoICNBjjz2miIgIpaWlKS0tzabfggULNGTIEM2cOVOzZ89Wp06ddN1112nNmjW1zvn+++/rjjvuUHx8vBYsWFDvwjClpaW69NJLtWfPHt17771auHChbrzxRn366aeNvscHHnhA4eHhmjRpko4cOSJJWrdunRYvXqzU1FSFhYVZY7voootUUVGhtLQ0zZ49W4cPH9bFF1+sgoKCE75eAADnIke3TI6W/phfddSoUTp48KBSUlI0Z84chYeHn/ANaUPi+frrr3XllVfKYrFo5syZevzxx/X3v//d5oPtxYsX61//+pdCQ0OVkZGhGTNmKDw8XJ999pm1T0lJic477zy99957Sk5O1oIFC9SvXz9NmjRJGRkZJ3iFAQB/1lFy7DXXXKM333xTiYmJeuqpp/Svf/1LR44cUVFRkbXPiy++qFGjRsnLy0uPPvqoHnzwQX3zzTe68MILaxXc/2ry5MlKTU3VueeeqyeeeELDhw9Xenq6xo0bV6vvrl27dO2112rkyJF6/PHH1b17d02cOLFBBXtJ+u2331RWVlZrO3r0aK2+7777rr777jslJiZq4cKFGjdunJYvX64rrrjC5kP6//u//9PTTz+ta665Rk899ZTuuusudenSRdu2bZMkffvtt9q+fbvGjh0rb2/vE8ZYUVGh5557TiNGjNCjjz6qhx56SAcOHFBcXJy2bNlSq/8LL7ygf//735o2bZpSUlK0detWXXzxxSopKTnhtTZs2KCpU6dq3Lhxeuyxx/Tbb7/pmmuuqfVhASADgMOsX7/ecHV1NVxdXY2YmBjj7rvvNtatW2dUVVXZ9Pv+++8NScbzzz9f6xySjLS0NOt+WlqaIckYP358rb4xMTFGRESETVtBQYEhyXjhhResbQkJCUbv3r2t+x9//LEhyXj55Zdtxubk5NTZ/lcJCQmGJOPWW2+1ttXU1BijRo0y3NzcjAMHDljbf/nlF5uxVVVVxqBBg4yLL7641n27uLgYX3/9td1rG4ZhvPnmm4YkY9OmTfX2acw9fvXVV4abm5sxefJk49ChQ0ZQUJARGRlp/P7779Z7O/PMM424uDijpqbG5t7OOOMMY+TIkSeMGQDgXOTolsnRhw8fNry9vY3o6Gjj119/tTn25xz61/tuaDxPPPGEIcnmPv5qzJgxxtlnn203zkmTJhm9evUyysrKbNrHjRtn+Pr61ooFAFC/jpBjDx06ZEgy5s6dW2+fI0eOGN26dTOSkpJs2ouLiw1fX1+b9uP3d9yWLVsMScbkyZNtxt51112GJOP999+3tvXu3duQZHz00UfWttLSUsPd3d248847643vOEkn3P78XruunPjqq6/WisHX19eYNm1avdddtWqVIcl44oknThijYRjGsWPHDIvFYtN26NAhIyAgwLjlllusbcd/rrp06WL88MMP1vbPPvvMkGTccccd1ra/vu6G8cfr4ebmZuzatcva9sUXXxiSjIULFzYoVnQcPJEOONDIkSOVn5+vv//97/riiy/02GOPKS4uTkFBQXr77bdP6tz/93//V6stPj5ehYWF2r17t7UtOztb7u7uGjNmTL3nWrFihXx9fTVy5EibT58jIiLk5eVl96tSf/bnlb6Pfx2qqqpK7733nrW9S5cu1n8fOnRI5eXlGjZsmM3Xu44bPny4QkNDT3jd418ZXL16db1PDTTmHgcNGqQZM2boueeeU1xcnMrKyrRs2TLr3GlbtmzRt99+qxtuuEE///yz9VyVlZW65JJL9NFHH6mmpuaEcQMAnIcc3TI5+t1339WRI0d077331ppD/a9fpf6rhsRz/G+AVatW1Zt7u3Xrph9++EGbNm2q87hhGHrjjTc0evRoGYZh8zrHxcWpvLy8ztcAAFC3jpBju3TpIjc3N+Xl5dU7Dcy7776rw4cPa/z48Tbnd3V1VXR0tN3zr127VpJkNptt2u+8805JqvVtsdDQUA0bNsy637NnTw0YMEDfffddvdf4szFjxujdd9+ttU2fPr3Oez/u+JPs5513niTVytGfffaZfvrppzqvWVFRIUkNehpdklxdXeXm5ibpjyl5Dh48qGPHjikyMrLOPD127FgFBQVZ96OiohQdHW19be2JjY1V3759rfuDBw+Wj49Pg19PdBwU0gEHGzp0qFauXKlDhw6poKBAKSkpOnLkiK699lp98803TT7vGWecUavtuuuuk4uLi7KzsyX98cZwxYoVuvzyy+Xj41Pvub799luVl5fL399fPXv2tNmOHj3aoAVhXFxc1KdPH5u2/v37S5LNV9ZWr16t8847Tx4eHurRo4d69uypp59+WuXl5Q26x7oMHz5c11xzjWbMmCE/Pz+NGTNGzz//vM3ccY29x+nTpyssLEwFBQVKS0uzKRZ8++23kqSEhIRa53ruuedksVjqvB8AQOtCjm7+HH28qDFo0KAG9f+zhsQTHx+vCy64QJMnT1ZAQIDGjRun1157zaaofs8998jLy0tRUVE688wzNW3aNJupXw4cOKDDhw/r2WefrfUaJyYmSnLs4ngA0BG09xzr7u6uRx99VO+8844CAgJ00UUX6bHHHlNxcbHN+SXp4osvrnX+9evX2z3/3r175eLion79+tm0BwYGqlu3brXmhD/99NNrnaN79+4Nmutdkk477TTFxsbW2ur60PzgwYO67bbbFBAQoC5duqhnz57W/y5/ztGPPfaYtm7dquDgYEVFRemhhx6yKUQf/29zfErVhli2bJkGDx4sDw8PnXLKKerZs6fWrFlT598qZ555Zq22/v37n3BKHenkX090HCxVCzQTNzc3DR06VEOHDlX//v2VmJioFStWKC0trd4nsqqrq+s9358/BT7u1FNP1bBhw/Taa6/pvvvu06effqqioiI9+uijdmOrqamRv7+/Xn755TqP9+zZ0+74hvr444/197//XRdddJGeeuop9erVS507d9bzzz9f54Jjdd1jXUwmk15//XV9+umn+s9//qN169bplltu0eOPP65PP/1UXl5ejb7H7777zvqHz1dffWVz7Pib87lz5yo8PLzO83l5eTUodgCA85Gjmy9HN3c8Xbp00UcffaQPPvhAa9asUU5OjrKzs3XxxRdr/fr1cnV11VlnnaUdO3Zo9erVysnJ0RtvvKGnnnpKqampmjFjhjWv33TTTUpISKgznsGDBzfr/QJAe9Wec+ztt9+u0aNH66233tK6dev04IMPKj09Xe+//76GDBlizS8vvviiAgMDa40//o1ne0707a3jXF1d62w3/rKwuCNcf/312rhxo6ZPn67w8HDr++3LLrvM5oPs66+/XsOGDdObb76p9evXa+7cuXr00Ue1cuVKXX755Ro4cKCk2u+36/PSSy9p4sSJGjt2rKZPny5/f3+5uroqPT3d5tsIjtCSryfaNgrpQAuIjIyUJO3fv1+SrIufHD582KZfXSuPn0h8fLymTp2qHTt2KDs7W56enho9erTdMX379tV7772nCy64oMlvjGtqavTdd99Zn3CTpJ07d0qSdQGyN954Qx4eHlq3bp3c3d2t/Z5//vkmXfOvzjvvPJ133nl65JFH9Morr+jGG2/U8uXLNXny5EbdY01NjSZOnCgfHx/dfvvtmj17tq699lpdffXVkmT9ipePj49iY2MdEjsAoHUgRzs2Rx/PmVu3bq31VJ09jYnHxcVFl1xyiS655BLNnz9fs2fP1v33368PPvjAmqe7du2q+Ph4xcfHq6qqSldffbUeeeQRpaSkqGfPnvL29lZ1dTV5HQCaUXvMsX379tWdd96pO++8U99++63Cw8P1+OOP66WXXrLmQH9//0bnl969e6umpkbffvutzjrrLGt7SUmJDh8+rN69ezcp3pN16NAh5ebmasaMGUpNTbW2H38I7a969eqlqVOnaurUqSotLdW5556rRx55RJdffrn69++vAQMGaNWqVVqwYMEJH0Z7/fXX1adPH61cudLmA4a/Lp5uL6adO3faXSAdaCymdgEc6IMPPqjzE8vjc3INGDBA0h8FWT8/P3300Uc2/Z566qlGX/Oaa66Rq6urXn31Va1YsUJXXnmlunbtanfM9ddfr+rqas2aNavWsWPHjtX6w6Y+Tz75pPXfhmHoySefVOfOnXXJJZdI+uNTXZPJZPOEwZ49e/TWW2816Pz1OXToUK3X+fiT4send2nMPc6fP18bN27Us88+q1mzZun888/XlClTVFZWJkmKiIhQ3759NW/evDpXMT9w4MBJ3Q8AoPmRo1smR1966aXy9vZWenq6fvvtN5tj9p7qamg8Bw8erDX2r38D/PzzzzbH3dzcFBoaKsMw9Pvvv8vV1VXXXHON3njjDW3durXW+cjrANA4HSHH/vLLL7XyWt++feXt7W3NP3FxcfLx8dHs2bPrXMvLXn654oorJEkZGRk27fPnz5ckjRo1qt6xzen4k9p//e/71zirq6trTbfi7++vU0891WYK1hkzZujnn3/W5MmTdezYsVrXW79+vVavXl3vtT/77DPl5+fXGetbb72lH3/80bpfUFCgzz77TJdffvmJbhNoMJ5IBxzo1ltv1S+//KKrrrpKAwcOVFVVlTZu3Kjs7GyFhIRY592UpMmTJ2vOnDmaPHmyIiMj9dFHH1mfFmsMf39//e1vf9P8+fN15MgRxcfHn3DM8OHD9c9//lPp6enasmWLLr30UnXu3FnffvutVqxYoQULFujaa6+1ew4PDw/l5OQoISFB0dHReuedd7RmzRrdd9991q/EjRo1SvPnz9dll12mG264QaWlpVq0aJH69eunL7/8stH3etyyZcv01FNP6aqrrlLfvn115MgRLV68WD4+PtY/QBp6j9u2bdODDz6oiRMnWp9gWLp0qcLDwzV16lS99tprcnFx0XPPPafLL79cZ599thITExUUFKQff/xRH3zwgXx8fPSf//ynyfcDAGh+5OiWydE+Pj564oknNHnyZA0dOlQ33HCDunfvri+++EK//PKLli1bVue4hsYzc+ZMffTRRxo1apR69+6t0tJSPfXUUzrttNN04YUXSvqjmB8YGKgLLrhAAQEB2rZtm5588kmNGjXKusDZnDlz9MEHHyg6OlpJSUkKDQ3VwYMHtXnzZr333nt1FuwBAHXrCDl2586duuSSS3T99dcrNDRUnTp10ptvvqmSkhKNGzdO0h858Omnn9bNN9+sc889V+PGjVPPnj1VVFSkNWvW6IILLrD5oPvPwsLClJCQoGeffVaHDx/W8OHDVVBQoGXLlmns2LH629/+1ujXyBF8fHys88H//vvvCgoK0vr16/X999/b9Dty5IhOO+00XXvttQoLC5OXl5fee+89bdq0SY8//ri1X3x8vL766is98sgj+u9//6vx48erd+/e+vnnn5WTk6Pc3FzrlG5XXnmlVq5cqauuukqjRo3S999/r8zMTIWGhtb5gFu/fv104YUXasqUKbJYLMrIyNApp5yiu+++u3lfJHQsBgCHeeedd4xbbrnFGDhwoOHl5WW4ubkZ/fr1M2699VajpKTEpu8vv/xiTJo0yfD19TW8vb2N66+/3igtLTUkGWlpadZ+aWlphiTjwIED9V538eLFhiTD29vb+PXXX2sdT0hIMHr37l2r/dlnnzUiIiKMLl26GN7e3sY555xj3H333cZPP/1k9z4TEhKMrl27Grt37zYuvfRSw9PT0wgICDDS0tKM6upqm75LliwxzjzzTMPd3d0YOHCg8fzzz1vv6c8kGdOmTbN73eM2b95sjB8/3jj99NMNd3d3w9/f37jyyiuNzz//vFH3eOzYMWPo0KHGaaedZhw+fNhm3IIFCwxJRnZ2trXtv//9r3H11Vcbp5xyiuHu7m707t3buP76643c3NwGxQ0AcB5ydMvk6OPefvtt4/zzzze6dOli+Pj4GFFRUcarr75q974bEk9ubq4xZswY49RTTzXc3NyMU0891Rg/fryxc+dOa59nnnnGuOiii6z5um/fvsb06dON8vJym+uVlJQY06ZNM4KDg43OnTsbgYGBxiWXXGI8++yzjbpXAOjoOkKOLSsrM6ZNm2YMHDjQ6Nq1q+Hr62tER0cbr732Wq2+H3zwgREXF2f4+voaHh4eRt++fY2JEyfavF+tK9/+/vvvxowZM4wzzjjD6Ny5sxEcHGykpKQYv/32m02/3r17G6NGjap13eHDhxvDhw+v9x6Os5fXn3/+eUOSsWnTJmvbDz/8YFx11VVGt27dDF9fX+O6664zfvrpJ5v/ZhaLxZg+fboRFhZmeHt7G127djXCwsKMp556qs7rHM/n/v7+RqdOnYyePXsao0ePNlatWmXtU1NTY8yePdvo3bu34e7ubgwZMsRYvXp1rf+u33//vSHJmDt3rvH4448bwcHBhru7uzFs2DDjiy++sLluY/7O6d27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lZaW1tm/R48euv/++5Wfn68vv/xSiYmJSkxM1Lp16xxxiwAAtCu87wYAoHVqVCF95syZ+vrrr637X331lSZNmqTY2Fjde++9+s9//qP09HSHBwkAAJqmOXL3/PnzlZSUpMTERIWGhiozM1Oenp71fs18xIgRuuqqq3TWWWepb9++uu222zR48GBt2LDhpO4NAID2iPfdAAC0To0qpG/ZskWXXHKJdX/58uWKjo7W4sWLZTab9e9//1uvvfaaw4MEAABN4+jcXVVVpcLCQsXGxlrbXFxcFBsbq/z8/BOONwxDubm52rFjhy666KI6+1gsFlVUVNhsAAB0FLzvBgCgdWpUIf3QoUMKCAiw7n/44Ye6/PLLrftDhw7Vvn37HBcdAAA4KY7O3WVlZaqurrY5pyQFBASouLi43nHl5eXy8vKSm5ubRo0apYULF2rkyJF19k1PT5evr691Cw4ObnB8AAC0dbzvBgCgdWpUIT0gIMC64ElVVZU2b96s8847z3r8yJEj6ty5s2MjBAAATdZacre3t7e2bNmiTZs26ZFHHpHZbFZeXl6dfVNSUlReXm7dKBYAADqS1pK7AQCArUYV0q+44grde++9+vjjj5WSkiJPT08NGzbMevzLL79U3759HR4kAABoGkfnbj8/P7m6uqqkpMSmvaSkRIGBgfWOc3FxUb9+/RQeHq4777xT1157bb3zu7q7u8vHx8dmAwCgo2iu992LFi1SSEiIPDw8FB0drYKCgnr7rly5UpGRkerWrZu6du2q8PBwvfjii026HwAA2otGFdJnzZqlTp06afjw4Vq8eLEWL14sNzc36/GsrCxdeumlDg8SAAA0jaNzt5ubmyIiIpSbm2ttq6mpUW5urmJiYhp8npqaGlkslgb3BwCgo2iO993Z2dkym81KS0vT5s2bFRYWpri4OJWWltbZv0ePHrr//vuVn5+vL7/8UomJiUpMTNS6detO6t4AAGjLOjWms5+fnz766CPrPKeurq42x1esWCEvLy+HBggAAJquOXK32WxWQkKCIiMjFRUVpYyMDFVWVioxMVGSNGHCBAUFBVmfOE9PT1dkZKT69u0ri8WitWvX6sUXX9TTTz/tmJsEAKAdaY7cPX/+fCUlJVlzdWZmptasWaOsrCzde++9tfqPGDHCZv+2227TsmXLtGHDBsXFxTXuhgAAaCcaVUg/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tHjxYlVWVioxMVHe3t7q16+f9WmW6upqTZo0SWeccYa6dOmiAQMGaMGCBXavUVNTo/T0dOuYsLAwvf766y1xe0CzmThxom699VYVFRXJZDIpJCTkhD/reXl5MplMWrdunYYMGaIuXbro4osvVmlpqd555x2dddZZ8vHx0Q033KBffvnFOi4nJ0cXXnihunXrplNOOUVXXnmldu/ebTe+rVu36vLLL5eXl5cCAgJ08803q6ysrNleDwC1XX755Xr44Yd11VVXOTsUtCE5OTmaOHGizj77bIWFhWnp0qUqKipSYWGhs0NDKzd69GhdccUVOvPMM9W/f3898sgj8vLy0qeffurs0AA00fz585WUlKTExESFhoYqMzNTnp6eysrKcnZoQJvF3+jtG4V0NLtly5bJz89PBQUFuvXWWzVlyhRdd911Ov/887V582Zdeumluvnmm/XLL7+opqZGp512mlasWKFvvvlGqampuu+++/Taa6/Ve/709HS98MILyszM1Ndff6077rhDN910kz788MMWvEvAsRYsWKCZM2fqtNNO0/79+7Vp06YG/6w/9NBDevLJJ7Vx40bt27dP119/vTIyMvTKK69ozZo1Wr9+vRYuXGjtX1lZKbPZrM8//1y5ublycXHRVVddpZqamjpjO3z4sC6++GINGTJEn3/+uXJyclRSUqLrr7++WV8TAIDjlZeXS5J69Ojh5EjQllRXV2v58uWqrKxUTEyMs8MB0ARVVVUqLCxUbGystc3FxUWxsbHKz893YmQA0HoxTwCaXVhYmB544AFJUkpKiubMmSM/Pz8lJSVJklJTU/X000/ryy+/1HnnnacZM2ZYx55xxhnKz8/Xa6+9VmeRzmKxaPbs2Xrvvfesf8T36dNHGzZs0DPPPKPhw4e3wB0Cjufr6ytvb2+5uroqMDCwUT/rDz/8sC644AJJ0qRJk5SSkqLdu3erT58+kqRrr71WH3zwge655x5J0jXXXGNz7aysLPXs2VPffPNNnfOyP/nkkxoyZIhmz55tMyY4OFg7d+5U//79HftiAACaRU1NjW6//XZdcMEFrMOBBvnqq68UExOj3377TV5eXnrzzTcVGhrq7LAANEFZWZmqq6sVEBBg0x4QEKDt27c7KSoAaN0opKPZDR482PpvV1dXnXLKKTrnnHOsbccT9/F52BYtWqSsrCwVFRXp119/VVVVlcLDw+s8965du/TLL79o5MiRNu1VVVUaMmSIg+8EcJ7G/Kz/+XcuICBAnp6e1iL68bY/LyL07bffKjU1VZ999pnKysqsT6IXFRXVWVj54osv9MEHH8jLy6vWsd27d1NIB4A2Ytq0adq6das2bNjg7FDQRgwYMEBbtmxReXm5Xn/9dSUkJOjDDz+kmA4AADoECulodp07d7bZN5lMNm0mk0nSH09FLV++XHfddZcef/xxxcTEyNvbW3PnztVnn31W57mPHj0qSVqzZo2CgoJsjrm7uzvyNgCnaszP+l9/v+r6HfzztC2jR49W7969tXjxYp166qmqqanRoEGDVFVVVW8so0eP1qOPPlrrWK9evRp3YwAAp0hOTtbq1av10Ucf6bTTTnN2OGgj3Nzc1K9fP0lSRESENm3apAULFuiZZ55xcmQAGsvPz0+urq4qKSmxaS8pKVFgYKCTogKA1o1COlqVTz75ROeff76mTp1qbbO36GFoaKjc3d1VVFTENC5o15rrZ/3nn3/Wjh07tHjxYg0bNkySTvhk4rnnnqs33nhDISEh6tSJNAIAbYlhGLr11lv15ptvKi8vT2eccYazQ0IbVlNTI4vF4uwwADSBm5ubIiIilJubq7Fjx0r643c6NzdXycnJzg0OAFopKiBoVc4880y98MILWrdunc444wy9+OKL2rRpU71v8ry9vXXXXXfpjjvuUE1NjS688EKVl5frk08+kY+PjxISElr4DoDm0Vw/6927d9cpp5yiZ599Vr169VJRUZHuvfdeu2OmTZumxYsXa/z48br77rvVo0cP7dq1S8uXL9dzzz0nV1fXJsUCoHGOHj2qXbt2Wfe///57bdmyRT169NDpp5/uxMjQmk2bNk2vvPKKVq1aJW9vbxUXF0v6Y22OLl26ODk6tGYpKSm6/PLLdfrpp+vIkSN65ZVXlJeXp3Xr1jk7NABNZDablZCQoMjISEVFRSkjI0OVlZVKTEx0dmhAm8Xf6O0bhXS0Kv/85z/13//+V/Hx8TKZTBo/frymTp2qd955p94xs2bNUs+ePZWenq7vvvtO3bp107nnnqv77ruvBSMHml9z/Ky7uLho+fLl+te//qVBgwZpwIAB+ve//60RI0bUO+bUU0/VJ598onvuuUeXXnqpLBaLevfurcsuu0wuLi5NjgVA43z++ef629/+Zt03m82SpISEBC1dutRJUaG1e/rppyWp1v/nn3/+eU2cOLHlA0KbUVpaqgkTJmj//v3y9fXV4MGDtW7dulrrtwBoO+Lj43XgwAGlpqaquLhY4eHhysnJqbUAKYCG42/09s1kGIbh7CAAAAAAAAAAAGiteHQQAAAAAAAAAAA7KKQDAAAAAAAAAGAHhXQAAAAAAAAAAOygkA4AAAAAAAAAgB0U0gEAAAAAAAAAsINCOgAAAAAAAAAAdlBIBwAAAAAAAADADgrpAAAAAAAAAADYQSEdAAAA6ABGjBih22+/3dlhAACAFpKXlyeTyaTDhw/X22fp0qXq1q1bi8UEtGUU0gHYdeDAAU2ZMkWnn3663N3dFRgYqLi4OH3yySfODg0AgA5n4sSJMplMMplMcnNzU79+/TRz5kwdO3bM2aEBANAhTJw4UWPHjq3V3pCidWNUVFTo/vvv18CBA+Xh4aHAwEDFxsZq5cqVMgzDIdeQpPj4eO3cudNh5wPas07ODgBA63bNNdeoqqpKy5YtU58+fVRSUqLc3Fz9/PPPzg4NAIAO6bLLLtPzzz8vi8WitWvXatq0aercubNSUlKcHRoAAHCAw4cP68ILL1R5ebkefvhhDR06VJ06ddKHH36ou+++WxdffLHDniLv0qWLunTp4pBzAe0dT6QDqNfhw4f18ccf69FHH9Xf/vY39e7dW1FRUUpJSdHf//53a5/JkyerZ8+e8vHx0cUXX6wvvvhC0h9PswcGBmr27NnWc27cuFFubm7Kzc11yj0BANDWHf+GWO/evTVlyhTFxsbq7bffliR98sknGjFihDw9PdW9e3fFxcXp0KFDdZ7nxRdfVGRkpLy9vRUYGKgbbrhBpaWl1uOHDh3SjTfeqJ49e6pLly4688wz9fzzz0uSqqqqlJycrF69esnDw0O9e/dWenp68988AABtwM8//6zx48crKChInp6eOuecc/Tqq6/a9Hn99dd1zjnnqEuXLjrllFMUGxuryspKSdJ9992nPXv26LPPPlNCQoJCQ0PVv39/JSUlacuWLfLy8pJ04lx+3CeffKLBgwfLw8ND5513nrZu3Wo99tepXR566CGFh4frxRdfVEhIiHx9fTVu3DgdOXKkGV4poG2hkA6gXl5eXvLy8tJbb70li8VSZ5/rrrtOpaWleuedd1RYWKhzzz1Xl1xyiQ4ePKiePXsqKytLDz30kD7//HMdOXJEN998s5KTk3XJJZe08N0AANA+denSRVVVVdqyZYsuueQShYaGKj8/Xxs2bNDo0aNVXV1d57jff/9ds2bN0hdffKG33npLe/bs0cSJE63HH3zwQX3zzTd65513tG3bNj399NPy8/OTJP373//W22+/rddee007duzQyy+/rJCQkBa4WwAAWr/ffvtNERERWrNmjbZu3ap//OMfuvnmm1VQUCBJ2r9/v8aPH69bbrlF27ZtU15enq6++moZhqGamhotX75cN954o0499dRa5/by8lKnTn9MMHGiXH7c9OnT9fjjj2vTpk3q2bOnRo8erd9//73e+Hfv3q233npLq1ev1urVq/Xhhx9qzpw5jnlxgDaMqV0A1KtTp05aunSpkpKSlJmZqXPPPVfDhw/XuHHjNHjwYG3YsEEFBQUqLS2Vu7u7JGnevHl666239Prrr+sf//iHrrjiCiUlJenGG29UZGSkunbtyhNrAAA4gGEYys3N1bp163TrrbfqscceU2RkpJ566ilrn7PPPrve8bfccov133369NG///1vDR06VEePHpWXl5eKioo0ZMgQRUZGSpJNobyoqEhnnnmmLrzwQplMJvXu3dvxNwgAQCu1evVq61Phx/35g+ugoCDddddd1v1bb71V69at02uvvaaoqCjt379fx44d09VXX23Noeecc44kqbS0VIcOHdLAgQNPGMeJcvlxaWlpGjlypCRp2bJlOu200/Tmm2/q+uuvr/O8NTU1Wrp0qby9vSVJN998s3Jzc/XII4+cMCagPeOJdAB2XXPNNfrpp5/09ttv67LLLlNeXp7OPfdcLV26VF988YWOHj2qU045xfr0upeXl77//nvt3r3beo558+bp2LFjWrFihV5++WVr0R0AADTe8TfvHh4euvzyyxUfH6+HHnrI+kR6QxUWFmr06NE6/fTT5e3treHDh0v6o0guSVOmTNHy5csVHh6uu+++Wxs3brSOnThxorZs2aIBAwboX//6l9avX+/YmwQAoBX729/+pi1btthszz33nPV4dXW1Zs2apXPOOUc9evSQl5eX1q1bZ82xYWFhuuSSS3TOOefouuuu0+LFi61TsTVmIdET5fLjYmJirP/u0aOHBgwYoG3bttV73pCQEGsRXZJ69epV55QxQEdDIR3ACXl4eGjkyJF68MEHtXHjRk2cOFFpaWk6evSoevXqVesPiB07dmj69OnW8bt379ZPP/2kmpoa7dmzx3k3AgBAO3D8zfu3336rX3/9VcuWLVPXrl0btVBYZWWl4uLi5OPjo5dfflmbNm3Sm2++KemP+c8l6fLLL9fevXt1xx136KefftIll1xifbru3HPP1ffff69Zs2bp119/1fXXX69rr73W8TcLAEAr1LVrV/Xr189mCwoKsh6fO3euFixYoHvuuUcffPCBtmzZori4OGuOdXV11bvvvqt33nlHoaGhWrhwoQYMGKDvv/9ePXv2VLdu3bR9+3a7MTQklzdV586dbfZNJpNqampO6pxAe0AhHUCjhYaGqrKyUueee66Ki4vVqVOnWn9EHJ9DtaqqSjfddJPi4+M1a9YsTZ48mU+yAQA4CcffvJ9++unWOVIlafDgwQ1ezHv79u36+eefNWfOHA0bNkwDBw6sMz/37NlTCQkJeumll5SRkaFnn33WeszHx0fx8fFavHixsrOz9cYbb+jgwYMnf4MAALRxn3zyicaMGaObbrpJYWFh6tOnj3bu3GnTx2Qy6YILLtCMGTP03//+V25ubnrzzTfl4uKicePG6eWXX9ZPP/1U69xHjx7VsWPHGpzLJenTTz+1/vvQoUPauXOnzjrrLMfeNNABUEgHUK+ff/5ZF198sV566SV9+eWX+v7777VixQo99thjGjNmjGJjYxUTE6OxY8dq/fr12rNnjzZu3Kj7779fn3/+uSTp/vvvV3l5uf7973/rnnvuUf/+/W3mcQMAAI6RkpKiTZs2aerUqfryyy+1fft2Pf300yorK6vV9/TTT5ebm5sWLlyo7777Tm+//bZmzZpl0yc1NVWrVq3Srl279PXXX2v16tXWN93z58/Xq6++qu3bt2vnzp1asWKFAgMD1a1bt5a4VQAAWrUzzzxT7777rjZu3Kht27bpn//8p0pKSqzHP/vsM82ePVuff/65ioqKtHLlSh04cMCaZx955BEFBwcrOjpaL7zwgr755ht9++23ysrK0pAhQ3T06NEG5fLjZs6cqdzcXG3dulUTJ06Un5+fxo4d2xIvBdCuUEgHUC8vLy9FR0friSee0EUXXaRBgwbpwQcfVFJSkp588kmZTCatXbtWF110kRITE9W/f3+NGzdOe/fuVUBAgPLy8pSRkaEXX3xRPj4+cnFx0YsvvqiPP/5YTz/9tLNvDwCAdqV///5av369vvjiC0VFRSkmJkarVq2yeWr9uJ49e2rp0qVasWKFQkNDNWfOHM2bN8+mj5ubm1JSUjR48GBddNFFcnV11fLlyyVJ3t7e1sVNhw4dqj179mjt2rVyceHtBQAADzzwgM4991zFxcVpxIgRCgwMtClc+/j46KOPPtIVV1yh/v3764EHHtDjjz+uyy+/XNIf85h/+umnuummm/Twww9ryJAhGjZsmF599VXNnTtXvr6+Dcrlx82ZM0e33XabIiIiVFxcrP/85z9yc3NriZcCaFdMRmNWMQAAAAAAAAAAoIPhkREAAAAAAAAAAOygkA4AAAAAAAAAgB0U0gEAAAAAAAAAsINCOgAAAAAAAAAAdlBIBwAAAAAAAADADgrpAAAAAAAAAADYQSEdAAAAAAAAAAA7KKQDAAAAAAAAAGAHhXQAAAAAAAAAAOygkA4AAAAAAAAAgB0U0gEAAAAAAAAAsINCOgAAAAAAAAAAdvw/S3H1p5xXNzEAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.histplot(data=df, x=\"Age\", hue=\"Survived\", bins=30, kde=True)\n",
        "plt.title(\"Distribution de l'âge selon la survie\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "cW91cFzgA78g",
        "outputId": "c17a3fe0-068a-46da-efd0-6adb51e79e33"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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8cdrnfvFF1+g0Wi46qqrTnm95visCi9RhGjBPvzwQwVQNmzYcMpzQkJClD59+nheP/7448rxH+3XXntNAZSioqJTXmPDhg0KoHz44YcnHbvwwgsVQJk/f36dxy688ELP6xUrViiA0rZtW8VkMnn2f/nllwqgvP7665597dq1U2bMmHHGa54uthkzZijt2rXzvP7mm28UQHnmmWdqnXfllVcqKpVK2bdvn2cfoOh0ulr7tm7dqgDKm2++edK9jjdv3jwFUL788kvPvqqqKqVDhw4KoKxYsUJRFEVxu91Kx44dlbFjxyput9tzrtlsVpKTk5UxY8ac9j5ZWVmnfPbjmc3mk/aNHTtWad++ved1fn6+otVqlcmTJ9c674knnlCAWu/F008/rQQEBCh79+6tde5DDz2kaDQaJTs7+5SxLF68+Iyf2VWrVimA8umnn9bav3Tp0pP2n/h5qPnZf/LJJ559drtdGTJkiBIYGOj53NX87CIiIpSSkhLPud9++60CKN9///0p41OU/32Wa95LRan75zx37lxFpVIphw4dOu31evXqpUyYMOG055z4rDVO/JzXPFtwcLBSWFhY69x//vOfCqBs37691v6uXbsqF198sef1ic93rp9V0bJJM5Zo9QIDA087Kis0NBSAb7/99qyaIOqi1+uZOXPmWZ//t7/9jaCgIM/rK6+8ktjYWH766acG3f9s/fTTT2g0Gu68885a+++77z4URWHJkiW19o8ePZqUlBTP6549exIcHMyBAwfOeJ/Y2FiuvPJKzz6j0cgtt9xS67wtW7aQmZnJX/7yF44ePUpxcTHFxcVUVVUxatQoVq5c2eD35HjH15aVl5dTXFzMhRdeyIEDBygvLwcgLS0Np9PJP/7xj1plazq8H2/RokVccMEFhIWFeWIuLi5m9OjRuFwuVq5cecpYaj5vP/zwAw6Ho85zFi1aREhICGPGjKl1/X79+hEYGMiKFStOef2ffvqJmJgYrr32Ws8+Pz8/7rzzTiorK/n9999rnX/NNdcQFhbmeV1TO3im97gux/+cq6qqKC4uZujQoSiKwubNm09bNjQ0lJ07d5KZmVnv+57K1KlTadOmTa19U6ZMQavV8sUXX3j27dixg127dnHNNdec8lrN9VkV3iHNWKLVq6ysJCoq6pTHr7nmGv71r39x00038dBDDzFq1CimTJnClVdeiVp9dvl+27Zt69UZuWPHjrVeq1QqOnTowMGDB8/6Gg1x6NAh4uLiaiVaUN0cVnP8eImJiSddIyws7Iz9Rg4dOkSHDh1O6hvVuXPnWq9rvthmzJhxymuVl5fX+jJuiNWrV/P444+zZs0azGbzSdcPCQnxPHuHDh1qHQ8PDz/p/pmZmWzbtu2kL9IahYWFp4zlwgsvZOrUqTz55JO89tprjBw5ksmTJ/OXv/wFvV7vuX55efkpP7enu/6hQ4fo2LHjSZ/ds32Pa571bPoGnSg7O5vHHnuM77777qTyNUnlqTz11FNMmjSJTp060b17d8aNG8df//pXevbsWe84aiQnJ5+0LzIyklGjRvHll1/y9NNPA9VNWFqtlilTppzyWs31WRXeIcmOaNUOHz5MeXn5SV9gxzMYDKxcuZIVK1bw448/snTpUr744gsuvvhifv75ZzQazRnvU59+NmfrVBMfulyus4qpMZzqPsoJnZkbquYv4ZdeeonevXvXeU5gYOA53WP//v2MGjWKLl268Oqrr5KQkIBOp+Onn37itddea9Bf4263mzFjxvDAAw/UebxTp06nLFszqeTatWv5/vvvWbZsGTfccAOvvPIKa9euJTAwELfbTVRUFJ9++mmd1zhVktUQjfUeu1wuxowZQ0lJCQ8++CBdunQhICCAI0eOcP3115/x5zxixAj279/Pt99+y88//8y//vUvXnvtNebPn89NN90EVP/s6orL5XLVec1T/X85bdo0Zs6cyZYtW+jduzdffvklo0aNIjIy8pTxNcdnVXiPJDuiVfv4448BGDt27GnPU6vVjBo1ilGjRvHqq6/y3HPP8cgjj7BixQpGjx7d6DMun1hVrygK+/btq/VXbFhYGGVlZSeVPXToEO3bt/e8rk9s7dq149dff6WioqJW7c7u3bs9xxtDu3bt2LFjB4qi1IrvxDl/aprIgoODGT16dKPc+0Tff/89NpuN7777rlYtxolNQTXPvm/fvlo1AkePHj2pliIlJYXKyspzinnw4MEMHjyYZ599loULFzJ9+nQ+//xzbrrpJlJSUvj1118ZNmxYvRPpdu3asW3bNtxud63ancZ+j0+0fft29u7dy0cffcTf/vY3z/7jR1OdSXh4ODNnzmTmzJlUVlYyYsQInnjiCU+yExYWVmfz2om1VWcyefJkbr31Vk9T1t69e5k9e/ZpyzTHZ1V4j/TZEa3W8uXLefrpp0lOTmb69OmnPK+kpOSkfTV/udUMlQ4ICACoM/loiP/85z+1+hF99dVX5OXlMX78eM++lJQU1q5di91u9+z74YcfThqiXp/YLr30Ulwul2d4bY3XXnsNlUpV6/7n4tJLLyU3N7fWshhms5n33nuv1nn9+vUjJSWFl19+mcrKypOuU1RUdM6x1NRcHF8jUF5ezocffljrvFGjRqHVak8akn7izwrg6quvZs2aNSxbtuykY2VlZTidzlPGU1paelLtxImft6uvvhqXy+VpZjme0+k87Xt96aWXkp+fX6tPitPp5M033yQwMJALL7zwlGXPRV0/Z0VReP3118+q/NGjR2u9DgwMpEOHDrWmK0hJSWH37t21Phdbt26t99Dv0NBQxo4dy5dffsnnn3+OTqdj8uTJpy3THJ9V4T1SsyNahSVLlrB7926cTicFBQUsX76cX375hXbt2vHdd9+ddnKwp556ipUrVzJhwgTatWtHYWEh77zzDvHx8QwfPhyo/iUbGhrK/PnzCQoKIiAggEGDBtXZJ+BshIeHM3z4cGbOnElBQQHz5s2jQ4cOtYbH33TTTXz11VeMGzeOq6++mv379/PJJ5/U6jBc39guv/xyLrroIh555BEOHjxIr169+Pnnn/n222+5++67T7p2Q91888289dZb/O1vfyM9PZ3Y2Fg+/vhjjEZjrfPUajX/+te/GD9+PN26dWPmzJm0bduWI0eOsGLFCoKDg/n+++/PKZZLLrkEnU7H5Zdfzq233kplZSXvv/8+UVFR5OXlec6Ljo7mrrvu4pVXXmHixImMGzeOrVu3smTJEiIjI2vVUN1///189913XHbZZVx//fX069ePqqoqtm/fzldffcXBgwdP2STy0Ucf8c4773DFFVeQkpJCRUUF77//PsHBwVx66aVAdb+eW2+9lblz57JlyxYuueQS/Pz8yMzMZNGiRbz++uu1On8f75ZbbuGf//wn119/Penp6SQlJfHVV1+xevVq5s2bd1J/rcbSpUsXUlJS+L//+z+OHDlCcHAw//3vf8+670/Xrl0ZOXIk/fr1Izw8nI0bN/LVV18xa9Yszzk33HADr776KmPHjuXGG2+ksLCQ+fPn061bN0wmU73iveaaa7juuut45513GDt2rKfj+Kk0x2dVeJGXRoEJcVZqhp7XbDqdTomJiVHGjBmjvP7667WGd9c4ceh5WlqaMmnSJCUuLk7R6XRKXFyccu211540rPjbb79Vunbtqmi12lrDnS+88EKlW7dudcZ3qqHnn332mTJ79mwlKipKMRgMyoQJE+ocmvvKK68obdu2VfR6vTJs2DBl48aNdQ6/PVVsJw7JVRRFqaioUO655x4lLi5O8fPzUzp27Ki89NJLtYbTKkr10PPbb7/9pJhONST+RIcOHVImTpyoGI1GJTIyUrnrrrs8Q6ePH66sKIqyefNmZcqUKUpERISi1+uVdu3aKVdffbWSlpZ22nuc7dDz7777TunZs6fi7++vJCUlKS+88ILywQcfKICSlZXlOc/pdCpz5sxRYmJiFIPBoFx88cVKRkaGEhERofz973+vdc2Kigpl9uzZSocOHRSdTqdERkYqQ4cOVV5++WXFbrefMpZNmzYp1157rZKYmKjo9XolKipKueyyy5SNGzeedO57772n9OvXTzEYDEpQUJDSo0cP5YEHHlByc3M959T1eSgoKFBmzpypREZGKjqdTunRo8dJP6Oan91LL7100n0B5fHHHz/1D1Spe+j5rl27lNGjRyuBgYFKZGSkcvPNN3umKzjTe/TMM88oAwcOVEJDQxWDwaB06dJFefbZZ0/6WX7yySdK+/btFZ1Op/Tu3VtZtmzZKYee1/VsNUwmk2IwGE4apn+651OUhn9WRcumUpRG6okohBCtUFlZGWFhYTzzzDO1lpwQQvgO6bMjhDhv1LWsQc1syHUtUyCE8A3SZ0cIcd744osvWLBgAZdeeimBgYH88ccffPbZZ1xyySUMGzbM2+EJIZqIJDtCiPNGz5490Wq1vPjii5hMJk+n5WeeecbboQkhmpD02RFCCCGET5M+O0IIIYTwaZLsCCGEEMKnebXPzsqVK3nppZdIT08nLy+PxYsXe2a5dDgcPProo/z0008cOHCAkJAQRo8ezfPPP09cXJznGiUlJdxxxx18//33qNVqpk6dyuuvv16vNUzcbje5ubkEBQU1+rIBQgghhGgaiqJQUVFBXFzcaRd29mqyU1VVRa9evbjhhhtOWo3WbDazadMm5syZQ69evSgtLeWuu+5i4sSJbNy40XPe9OnTycvL45dffsHhcDBz5kxuueUWFi5ceNZx5ObmkpCQ0GjPJYQQQojmk5OTQ3x8/CmPt5gOyiqVqlbNTl02bNjAwIEDOXToEImJiWRkZNC1a1c2bNhA//79AVi6dCmXXnophw8frlUDdDybzVZrPZby8nISExPJyckhODi4UZ9LCCGEEE3DZDKRkJBAWVkZISEhpzyvVQ09Ly8vR6VSedY4WbNmDaGhoZ5EB2D06NGo1WrWrVvHFVdcUed15s6dy5NPPnnS/uDgYEl2hBBCiFbmTF1QWk0HZavVyoMPPsi1117rSUjy8/OJioqqdZ5WqyU8PJz8/PxTXmv27NmUl5d7thNXmRZCCCGE72gVNTsOh4Orr74aRVF49913z/l6er0evV7fCJEJIYQQoqVr8clOTaJz6NAhli9fXquZKSYmhsLCwlrnO51OSkpKiImJae5QhRBCCNECtehkpybRyczMZMWKFURERNQ6PmTIEMrKykhPT6dfv34ALF++HLfbzaBBg7wRshBCCNEgbrcbu93u7TBaFD8/PzQazTlfx6vJTmVlJfv27fO8zsrKYsuWLYSHhxMbG8uVV17Jpk2b+OGHH3C5XJ5+OOHh4eh0OlJTUxk3bhw333wz8+fPx+FwMGvWLKZNm3bKkVhCCCFES2O328nKysLtdns7lBYnNDSUmJiYc5oHz6tDz3/77Tcuuuiik/bPmDGDJ554guTk5DrLrVixgpEjRwLVkwrOmjWr1qSCb7zxRr0mFTSZTISEhFBeXi6jsYQQQjQrRVHIzs7G4XCccXK884miKJjNZgoLCwkNDSU2Nvakc872+9urNTsjR47kdLnW2eRh4eHh9ZpAUAghhGhJnE4nZrOZuLg4jEajt8NpUQwGAwCFhYVERUU1uElL0kchhBDCi1wuFwA6nc7LkbRMNQmgw+Fo8DUk2RFCCCFaAFmbsW6N8XORZEcIIYQQPk2SHSGEEEKc5LfffkOlUlFWVtak97n++utPuy5mY5BkRwghhGjBioqKuO2220hMTESv1xMTE8PYsWNZvXp1k9536NCh5OXlnXaBzdaiRU8qKIQQQpzvpk6dit1u56OPPqJ9+/YUFBSQlpbG0aNHG3Q9RVFwuVxotadPAXQ6nc+sRiA1O0IIIUQLVVZWxqpVq3jhhRe46KKLaNeuHQMHDmT27NlMnDiRgwcPolKp2LJlS60yKpWK3377Dfhfc9SSJUvo168fer2eDz74AJVKxe7du2vd77XXXiMlJaVWubKyMkwmEwaDgSVLltQ6f/HixQQFBWE2mwHIycnh6quvJjQ0lPDwcCZNmsTBgwc957tcLu69915CQ0OJiIjggQceOKtpZs6V1OwI4SXZ2dkUFxc3uHxkZCSJiYmNGJEQoqUJDAwkMDCQb775hsGDB5/TItYPPfQQL7/8Mu3btycsLIz333+fTz/9lKefftpzzqeffspf/vKXk8oGBwdz2WWXsXDhQsaPH1/r/MmTJ2M0GnE4HIwdO5YhQ4awatUqtFotzzzzDOPGjWPbtm3odDpeeeUVFixYwAcffEBqaiqvvPIKixcv5uKLL27wc50VRSjl5eUKoJSXl3s7FHGeOHTokGIwGhWgwZvBaFQOHTrk7UcRQpwji8Wi7Nq1S7FYLHUe/+qrr5SwsDDF399fGTp0qDJ79mxl69atiqIoSlZWlgIomzdv9pxfWlqqAMqKFSsURVGUFStWKIDyzTff1Lrua6+9pqSkpHhe79mzRwGUjIyMWuVKS0sVRVGUxYsXK4GBgUpVVZWiKNXfnf7+/sqSJUsURVGUjz/+WOncubPidrs917TZbIrBYFCWLVumKIqixMbGKi+++KLnuMPhUOLj45VJkyY16Odztt/fUrMjhBcUFxdjMZuZ/uBLRCem1Lt8QfZ+Pn3hfoqLi6V2RwgfN3XqVCZMmMCqVatYu3YtS5Ys4cUXX+Rf//qXZ+mks9G/f/9ar6dNm8b//d//sXbtWgYPHsynn35K37596dKlS53lL730Uvz8/Pjuu++YNm0a//3vfwkODmb06NEAbN26lX379hEUFFSrnNVqZf/+/ZSXl5OXl1droW6tVkv//v2bvClLkh0hvCg6MYX4jt28HYYQooXz9/dnzJgxjBkzhjlz5nDTTTfx+OOPs2rVKqD28kqnmmk4ICCg1uuYmBguvvhiFi5cyODBg1m4cCG33XbbKWPQ6XRceeWVLFy4kGnTprFw4UKuueYaT0fnyspK+vXrx6effnpS2TZt2tT7mRuTdFAWQgghWpmuXbtSVVXlSSLy8vI8x47vrHwm06dP54svvmDNmjUcOHCAadOmnfH8pUuXsnPnTpYvX8706dM9x/r27UtmZiZRUVF06NCh1hYSEkJISAixsbGsW7fOU8bpdJKenn7W8TaUJDtCCCFEC3X06FEuvvhiPvnkE7Zt20ZWVhaLFi3ixRdfZNKkSRgMBgYPHszzzz9PRkYGv//+O48++uhZX3/KlClUVFRw2223cdFFFxEXF3fa80eMGEFMTAzTp08nOTm5VpPU9OnTiYyMZNKkSaxatYqsrCx+++037rzzTg4fPgzAXXfdxfPPP88333zD7t27+cc//tHkkxaCJDtCCCFEixUYGMigQYN47bXXGDFiBN27d2fOnDncfPPNvPXWWwB88MEHOJ1O+vXrx913380zzzxz1tcPCgri8ssvZ+vWrbVqaU5FpVJx7bXX1nm+0Whk5cqVJCYmMmXKFFJTU7nxxhuxWq0EBwcDcN999/HXv/6VGTNmMGTIEIKCgrjiiivq8RNpGJXS1L2CWgGTyURISAjl5eWeN0SIprRp0yb69evHvW9/3aA+O4czd/Lq7VNIT0+nb9++TRChEKK5WK1WsrKySE5Oxt/f39vhtDin+/mc7fe31OwIIYQQwqdJsiOEEEIInybJjhBCCCF8miQ7QgghhPBpkuwIIYQQwqdJsiOEEEIInybJjhBCCCF8miQ7QgghhPBpkuwIIYQQwqfJqudCCCFEK5OdnU1xcXGz3S8yMpLExMRmu19jk2RHCCGEaEWys7PpkpqKxWxutnsajEZ2Z2TUO+F5++23eemll8jPz6dXr168+eabDBw4sImiPDVJdoQQQohWpLi4GIvZzPQHXyI6MaXJ71eQvZ9PX7if4uLieiU7X3zxBffeey/z589n0KBBzJs3j7Fjx7Jnzx6ioqKaMOKTSbIjhBBCtELRiSkNWki4ubz66qvcfPPNzJw5E4D58+fz448/8sEHH/DQQw81ayzSQVkIIYQQjcput5Oens7o0aM9+9RqNaNHj2bNmjXNHo8kO0IIIYRoVMXFxbhcLqKjo2vtj46OJj8/v9njkWRHCCGEED5Nkh0hhBBCNKrIyEg0Gg0FBQW19hcUFBATE9Ps8UiyI4QQQohGpdPp6NevH2lpaZ59brebtLQ0hgwZ0uzxyGgsIYQQQjS6e++9lxkzZtC/f38GDhzIvHnzqKqq8ozOak6S7AghhBCtUEH2/hZ9n2uuuYaioiIee+wx8vPz6d27N0uXLj2p03JzkGRHCCGEaEUiIyMxGI18+sL9zXZPg9FIZGRkvcvNmjWLWbNmNUFE9SPJjhBCCNGKJCYmsjsjQ9bGqgdJdoQQQohWJjExsVUnH81NRmMJIYQQwqdJsiOEEEIInybJjhBCCCF8miQ7QgghhPBpkuwIIYQQwqdJsiOEEEIInybJjhBCCCF8msyzI4QQQrQy2dnZMqlgPUiyI4QQQrQi2dnZpKZ2wWy2NNs9jUYDGRm7zzrhWblyJS+99BLp6enk5eWxePFiJk+e3LRBnoYkO0IIIUQrUlxcjNls4ZOHryY1sU2T3y8ju4jrnvuS4uLis052qqqq6NWrFzfccANTpkxp4gjPTJIdIYQQohVKTWxD305tvR1GncaPH8/48eO9HYaHdFAWQgghhE/zarKzcuVKLr/8cuLi4lCpVHzzzTe1jiuKwmOPPUZsbCwGg4HRo0eTmZlZ65ySkhKmT59OcHAwoaGh3HjjjVRWVjbjUwghhBCiJfNqslPTpvf222/XefzFF1/kjTfeYP78+axbt46AgADGjh2L1Wr1nDN9+nR27tzJL7/8wg8//MDKlSu55ZZbmusRhBBCCNHCebXPzuna9BRFYd68eTz66KNMmjQJgP/85z9ER0fzzTffMG3aNDIyMli6dCkbNmygf//+ALz55ptceumlvPzyy8TFxTXbswghhBCiZWqxfXaysrLIz89n9OjRnn0hISEMGjSINWvWALBmzRpCQ0M9iQ7A6NGjUavVrFu37pTXttlsmEymWpsQQgghfFOLTXby8/MBiI6OrrU/Ojracyw/P5+oqKhax7VaLeHh4Z5z6jJ37lxCQkI8W0JCQiNHL4QQQpy/Kisr2bJlC1u2bAGqKzC2bNlCdna2V+I5L4eez549m3vvvdfz2mQyScIjhBCiVcnILmqx99m4cSMXXXSR53XNd+6MGTNYsGBBY4V21lpsshMTEwNAQUEBsbGxnv0FBQX07t3bc05hYWGtck6nk5KSEk/5uuj1evR6feMHLYQQQjSxyMhIjEYD1z33ZbPd02g0EBkZedbnjxw5EkVRmjCi+mmxyU5ycjIxMTGkpaV5khuTycS6deu47bbbABgyZAhlZWWkp6fTr18/AJYvX47b7WbQoEHeCl0IIYRoMomJiWRk7Ja1serBq8lOZWUl+/bt87yuadMLDw8nMTGRu+++m2eeeYaOHTuSnJzMnDlziIuL86yvkZqayrhx47j55puZP38+DoeDWbNmMW3aNBmJJYQQwmclJia26uSjuXk12TlTm94DDzxAVVUVt9xyC2VlZQwfPpylS5fi7+/vKfPpp58ya9YsRo0ahVqtZurUqbzxxhvN/ixCCCGEaJm8muycqU1PpVLx1FNP8dRTT53ynPDwcBYuXNgU4QkhhBDCB7TYoedCCCHE+aQldehtSRrj5yLJjhBCCOFFGo0GALvd7uVIWiaz2QyAn59fg6/RYkdjCSGEEOcDrVaL0WikqKgIPz8/1Gqph4DqGh2z2UxhYSGhoaGepLAhJNkRQgghvEilUhEbG0tWVhaHDh3ydjgtTmho6GnnzjsbkuwIIYQQXqbT6ejYsaM0ZZ3Az8/vnGp0akiyI4QQQrQAarW61tQqovFIw6AQQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ8myY4QQgghfJokO0IIIYTwaZLsCCGEEMKnSbIjhBBCCJ/WopMdl8vFnDlzSE5OxmAwkJKSwtNPP42iKJ5zFEXhscceIzY2FoPBwOjRo8nMzPRi1EIIIYRoSVp0svPCCy/w7rvv8tZbb5GRkcELL7zAiy++yJtvvuk558UXX+SNN95g/vz5rFu3joCAAMaOHYvVavVi5EIIIYRoKbTeDuB0/vzzTyZNmsSECRMASEpK4rPPPmP9+vVAda3OvHnzePTRR5k0aRIA//nPf4iOjuabb75h2rRpXotdCCGEEC1Di67ZGTp0KGlpaezduxeArVu38scffzB+/HgAsrKyyM/PZ/To0Z4yISEhDBo0iDVr1pzyujabDZPJVGsTQgghhG9q0TU7Dz30ECaTiS5duqDRaHC5XDz77LNMnz4dgPz8fACio6NrlYuOjvYcq8vcuXN58sknmy5wIYQQQrQYLbpm58svv+TTTz9l4cKFbNq0iY8++oiXX36Zjz766JyuO3v2bMrLyz1bTk5OI0UshBBCiJamRdfs3H///Tz00EOevjc9evTg0KFDzJ07lxkzZhATEwNAQUEBsbGxnnIFBQX07t37lNfV6/Xo9fomjV0IIYQQLUOLrtkxm82o1bVD1Gg0uN1uAJKTk4mJiSEtLc1z3GQysW7dOoYMGdKssQohhBCiZWrRNTuXX345zz77LImJiXTr1o3Nmzfz6quvcsMNNwCgUqm4++67eeaZZ+jYsSPJycnMmTOHuLg4Jk+e7N3ghRBCCNEitOhk580332TOnDn84x//oLCwkLi4OG699VYee+wxzzkPPPAAVVVV3HLLLZSVlTF8+HCWLl2Kv7+/FyMXQgghREvRopOdoKAg5s2bx7x58055jkql4qmnnuKpp55qvsCEEEII0Wq06D47QgghhBDnSpIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZIdIYQQQvg0SXaEEEII4dMalOy0b9+eo0ePnrS/rKyM9u3bn3NQQviyg8VVLNpVQcyMeXx/2I/P1mezbGc+hSart0MTQgifpG1IoYMHD+JyuU7ab7PZOHLkyDkHJYQvUhSFD1cf5LmfMnC6FfQxHbC7obDCRmGFjd35FXRvG8ywlEj8/TTeDlcIIXxGvZKd7777zvPfy5YtIyQkxPPa5XKRlpZGUlJSowUnhK+w2F3c/9VWftiWB0D3KB2/f/QSV838B8aoRPYWVLC3oJIdR0zkllqZ2q8tRl2D/hYRQghxgnr9Np08eTIAKpWKGTNm1Drm5+dHUlISr7zySqMFJ0RzyM7Opri4uEFlIyMjSUxMPO05brfCvV9uYcmOfLRqFY9MSKWnfwn97/uFMN1txLcJJKVNID3bWli6M58Ss53Fm48wpW88BqnhEUKIc1avZMftdgOQnJzMhg0biIyMbJKghGgu2dnZdElNxWI2N6i8wWhkd0bGaROe19MyWbIjHz+Nio9uGMjQlEg2bSo96by2YQam9G3LV+mHKa608+2WI1zVLwGNWtWg2IQQQlRrUD15VlZWY8chhFcUFxdjMZuZ/uBLRCem1KtsQfZ+Pn3hfoqLi0+Z7CzZnsfraZkAPHtFD4amnP4PhDCjjil9qhOeApONtQeOMqyD/FEhhBDnosGdAtLS0khLS6OwsNBT41Pjgw8+OOfAhGhO0YkpxHfs1qjXLK2y88g3OwC4cXgyV/dPOKtyEYF6RqVG8+P2PNIPlZIcGUBcqKFRYxNCiPNJg4aeP/nkk1xyySWkpaVRXFxMaWlprU0IAS8s3U1JlZ1O0YE8NL5Lvcp2iAokNTYIBfh5VwF2p/uMZYQQQtStQTU78+fPZ8GCBfz1r39t7HiE8AkbD5bw+YYcoLr5yk9T/78rLuzUhsOlFsotDjZllzK4fURjhymEEOeFBtXs2O12hg4d2tixCOETXG6FR481X13TP4EBSeENuo5eq+GCY/110g+VUmlzNlqMQghxPmlQsnPTTTexcOHCxo5FCJ/w0/Y8dudXEOyvrXfz1Yk6RAUSG+KP062w9sDJs5YLIYQ4swY1Y1mtVt577z1+/fVXevbsiZ+fX63jr776aqMEJ0Rr43IrvHFs9NVNF7QnLEB3TtdTqVQM7xDJovTD7Mo10TshlMhAfWOEKoQQ540GJTvbtm2jd+/eAOzYsaPWMZVK5gQR568ft+eRWVhJsL+W64clNco140INpLQJYH9RFeuzSri0R2yjXFcIIc4XDUp2VqxY0dhxCNHqnVirE+zvd4YSZ29QcgT7i6rYV1hJmdneaNcVQojzQYP67AghTvZrRgH7GrlWp0abID3tIowoQHq2TO8ghBD10aCanYsuuui0zVXLly9vcEAnOnLkCA8++CBLlizBbDbToUMHPvzwQ/r37w9UryT9+OOP8/7771NWVsawYcN499136dixY6PFIMTZWLD6IADXDW7XqLU6NQa0C+fQUTMZeRW0k5YsIYQ4aw2q2enduze9evXybF27dsVut7Np0yZ69OjRaMGVlpYybNgw/Pz8WLJkCbt27eKVV14hLCzMc86LL77IG2+8wfz581m3bh0BAQGMHTsWq9XaaHEIcSa7802sOXAUjVrFdYPbNck94kL9iQn2x+VW2FchC4QKIcTZalDNzmuvvVbn/ieeeILKyspzCuh4L7zwAgkJCXz44YeefcnJyZ7/VhSFefPm8eijjzJp0iQA/vOf/xAdHc0333zDtGnTGi0WIU7noz8PAjCuW0yTLe2gUqnonxTGD9vyyKpUo9Ke20gvIYQ4XzRqn53rrruuUdfF+u677+jfvz9XXXUVUVFR9OnTh/fff99zPCsri/z8fEaPHu3ZFxISwqBBg1izZs0pr2uz2TCZTLU2IRqqwuZm8eYjAI3eV+dEyZEBBPlrsbtVGDsPb9J7CSGEr2jUZGfNmjX4+/s32vUOHDjg6X+zbNkybrvtNu68804++ugjAPLz8wGIjo6uVS46OtpzrC5z584lJCTEsyUknN0CjULUJS3LjNXhpltcMP3bhZ25wDlQq1T0aBsCQFDfS5v0XkII4Ssa1Iw1ZcqUWq8VRSEvL4+NGzcyZ86cRgkMwO12079/f5577jkA+vTpw44dO5g/fz4zZsxo8HVnz57Nvffe63ltMpkk4RENtvygBYDpg9o1yzxT3eKCWbO/GH1cF/aXOujb5HcUQojWrUE1O8fXioSEhBAeHs7IkSP56aefePzxxxstuNjYWLp27VprX2pqKtnZ2QDExMQAUFBQUOucgoICz7G66PV6goODa21CNIQurjOHTU78/dRc3qt5hkgZdVrijdWroC/bV9Us9xRCiNasQTU7x3cYbkrDhg1jz549tfbt3buXdu2qR7skJycTExNDWlqaZ0Znk8nEunXruO2225olRnF+C+xR3V/s0u6xBDXBcPNTaR/oJsesYWW2BZPV0SRD3YUQwlc0KNmpkZ6eTkZGBgDdunWjT58+jRJUjXvuuYehQ4fy3HPPcfXVV7N+/Xree+893nvvPaB6dMrdd9/NM888Q8eOHUlOTmbOnDnExcUxefLkRo1FiBM53RCQeiEAV/Vv3mbQCL2CvfgQRLbjx215XDswsVnvL4QQrUmDkp3CwkKmTZvGb7/9RmhoKABlZWVcdNFFfP7557Rp06ZRghswYACLFy9m9uzZPPXUUyQnJzNv3jymT5/uOeeBBx6gqqqKW265hbKyMoYPH87SpUsbtaO0EHU5YlGj1huJDtAwKDm8We+tUkHV9jR0F93Aoo05kuwIIcRpNKjPzh133EFFRQU7d+6kpKSEkpISduzYgclk4s4772zUAC+77DK2b9+O1WolIyODm2++udZxlUrFU089RX5+PlarlV9//ZVOnTo1agxC1OVQZfX/PhclGVCrm38B3KqdK1CrYFN2GfsKG29+KyGE8DUNSnaWLl3KO++8Q2pqqmdf165defvtt1myZEmjBSdES1Vpc1Jkq05wRiY1zSSCZ+KqKqVPjB6Ar9IPeyUGIYRoDRqU7Ljdbvz8Tu4Q6efnh9vtPueghGjpMgsqABXWw7uICjinrm/n5OLk6kRr8ebDuNyK1+IQQoiWrEHJzsUXX8xdd91Fbm6uZ9+RI0e45557GDVqVKMFJ0RLtaegAgBzxkqvxtE/1p9Qox8FJhurMou8GosQQrRUDUp23nrrLUwmE0lJSaSkpJCSkkJycjImk4k333yzsWMUokUpM9spMNkAhardf3g1Fj+Nist7xgHw3ZbcM5wthBDnpwbVvyckJLBp0yZ+/fVXdu/eDVRP9nf8GlVC+Kq9BdWdgaP8FQ6Zy7wbDDCpdxwfrz3Esp35WB0u/P1kRXQhhDhevWp2li9fTteuXTGZTKhUKsaMGcMdd9zBHXfcwYABA+jWrRurVq1qqliF8DpFUdiTX92ElWBsGf3T+iaG0TbUQJXdxfLdhd4ORwghWpx6JTvz5s3j5ptvrnN5hZCQEG699VZeffXVRgtOiJampMpOidmORqWibQtJdtRqFZcdW6pCmrKEEOJk9Up2tm7dyrhx4055/JJLLiE9Pf2cgxKipdpfVL0WVUK4Ab8G9XhrGpN6tQVg+Z5CTFaHl6MRQoiWpV6/rgsKCuoccl5Dq9VSVCQjQoTv2l9U3V8nJSrQy5HUlhobRIeoQOxON8t25Hs7HCGEaFHqley0bduWHTt2nPL4tm3biI1tnpWfhWhuJouDwgobKqB9ZIC3w6lFpVIxsdexUVlbpSlLCCGOV69k59JLL2XOnDlYrdaTjlksFh5//HEuu+yyRgtOiJakplYnLtSAUee9iQRPpSbZ+XP/UYoqbF6ORgghWo56/cZ+9NFH+frrr+nUqROzZs2ic+fOAOzevZu3334bl8vFI4880iSBCuFtNf11Utq0rFqdGkmRAfSKD2Hr4XJ+2p7HjKFJ3g5JCCFahHolO9HR0fz555/cdtttzJ49G0Wpnp5epVIxduxY3n77baKjo5skUCG8yWx3cqTMArS8/jrHu7xXHFsPl/Pd1lxJdoQQ4ph618W3a9eOn376idLSUvbt24eiKHTs2JGwsLCmiE+IFuHAsVqdqCA9wf6n7qTvbZf3iuPZnzJIP1TK4VIz8WFGb4ckhBBe1+DBs2FhYQwYMICBAwdKoiN83r6aUVhtWm6tDkB0sD+DkyMA+H5rnpejEUKIlqEFzRQiRMtkc7o4XFLdhNWhBTdh1ZjYW0ZlCSHE8STZEeIMDhabcSkKYUY/wgN03g7njMZ3j0GrVpGRZ2JfYaW3wxFCCK+TZEeIM9jfSpqwaoQadQzvGAnAD9ukdkcIISTZEeI0nC43B48eG3LeCpqwalzWs7op64dteZ5Rk0IIcb6SZEeI08guMeNwKQTqtUQH6b0dzlm7pFs0Oo2afYWV7Cmo8HY4QgjhVZLsCHEaB4r/N5GgSqXycjRnL9jfjws7twHgBxmVJYQ4z0myI8QpKIrCwWPJTnILWwvrbFzWs3qduh+25UpTlhDivCbJjhCnUFRho8ruwk+jom2Ywdvh1Nvo1Gj8/dQcPGpmZ67J2+EIIYTXSLIjxCkcPGoGICHMiFbd+v5XCdBrubhLFADfy6gsIcR5rPX9BheimWS14iasGjWjsn6UUVlCiPOYJDtC1MFsd5JvsgKQFNF6k52LOkdh1Gk4XGphS06Zt8MRQgivkGRHiDocOtaE1SZQT6B/vdfLbTEMOg2jU6OB6jl3hBDifCTJjhB1qBmFlRTZ+lcNrxmV9eO2PNxuacoSQpx/JNkR4gRut8KhkuqandbchFXjws5tCNJryTdZ2Xio1NvhCCFEs5NkR4gT5JVbsTnd+PupiQnx93Y450yv1TCmW01TlozKEkKcfyTZEeIEWcfWwmoXEYC6Fc2afDqXHxuV9dP2fFzSlCWEOM9IsiPECTyzJvtAE1aNYR0iCTH4UVxpY92Bo94ORwghmpUkO0Icx2R1cLTKjgpoF9H6OyfX0GnVjOsWA8D3MipLCHGekWRHiOPU1OrEhvjj76fxcjSN67Je1aOylu7Iw+FyezkaIYRoPpLsCHGcLM+Qc99pwqoxpH0EEQE6Ss0O/twvTVlCiPOHJDtCHON0uTlcagF8Y8j5ibQaNeO6Vzdl/bBVRmUJIc4fkuwIcczhUgtOt0KgXktkoM7b4TSJy3tVj8patjMfu1OasoQQ5wdJdoQ4pmbIeVKkEZWPDDk/0YCkcKKC9JisTlZlFnk7HCGEaBaS7AgBKIpvDjk/kUat4tIe1R2VZa0sIcT5QpIdIYAKJ5isTjRqFQnhvjPkvC6XHxuV9cuuAqwOl5ejEUKIpifJjhBAnqX6f4X4MAN+Gt/+36JPQhhxIf5U2pz8tqfQ2+EIIUST8+3f6kKcpfxjyY4vjsI6kVqtYsKxldBlgkEhxPlAkh1x3lPpjBy1VXdITvbB+XXqctmxtbLSMgqosDq8HI0QQjQtSXbEec+Q3AcFFWFGP0IMft4Op1n0jA+hfZsArA43S7bnezscIYRoUpLsiPOeIWUAcP7U6gCoVCqu7BcPwFfph70cjRBCNC1JdsR5za0oGNr3A86P/jrHu6JPW1QqWH+whOyjZm+HI4QQTUaSHXFe21/iQBMQhlalEBdq8HY4zSo2xMDwDpEA/HeT1O4IIXyX1tsBCOFN6Xk2AELVVgoL6td3pbi4GICMjIx637chZRpTdnY2xcXF9At3sgr4bO0BLgivQH0WM0dHRkaSmJjY9EEKIUQjkWRHnNfWZlcCcHDLn7z3e0G9yjpKjgBw3XXXNfj+lZWVDS7bUNnZ2XRJTcViNqPS6omf9TGFGBl+xfXYcnacsbzBaGR3RoYkPEKIVkOSHXHeKqywcizXYUCMQo8LBtSr/No/HazKgBF/uZP+w0bWq2zG+t9Z8tHrWK3WepVrDMXFxVjMZqY/+BLRiSmkH9VwsAoG3fws/SNOP6NyQfZ+Pn3hfoqLiyXZEUK0Gq0q2Xn++eeZPXs2d911F/PmzQPAarVy33338fnnn2Oz2Rg7dizvvPMO0dHR3g1WtHi/7aleCNOWt5eoYDWxEcH1Kh9sqF4ZPSQ6nviO3epVtiB7f73ObwrRiSnEd+yGqo2Fg+mHybVqiW7fyednkBZCnH9azW+1DRs28M9//pOePXvW2n/PPffw/fffs2jRIn7//Xdyc3OZMmWKl6IUrcmK3dVLJVj2b/RyJN4VF+JPiMEPh0thX2HzN6sJIURTaxXJTmVlJdOnT+f9998nLCzMs7+8vJx///vfvPrqq1x88cX069ePDz/8kD///JO1a9d6MWLR0tmdblZlVncwtuzf4OVovEulUpEaGwRARp7Jy9EIIUTjaxXJzu23386ECRMYPXp0rf3p6ek4HI5a+7t06UJiYiJr1qw55fVsNhsmk6nWJs4vGw+WUGlzEqh1Y8/f5+1wvC41proJL6fUgkmWjxBC+JgWn+x8/vnnbNq0iblz5550LD8/H51OR2hoaK390dHR5Oefehjx3LlzCQkJ8WwJCQmNHbZo4ZYfa8LqFOQAFO8G0wIEG/yID6ueZygjV5J/IYRvadHJTk5ODnfddReffvop/v7+jXbd2bNnU15e7tlycnIa7dqidVixpybZcXo5kpajW1x17c6OXBNutySAQgjf0aJHY6Wnp1NYWEjfvn09+1wuFytXruStt95i2bJl2O12ysrKatXuFBQUEBMTc8rr6vV69Hp9U4YuWrDso2b2F1WhVavoGNi6m2wac0LDDlGBrNxbTKXNycGjVbRvE3iu4QkhRIvQopOdUaNGsX379lr7Zs6cSZcuXXjwwQdJSEjAz8+PtLQ0pk6dCsCePXvIzs5myJAh3ghZtALLd1dPHtg/KQx/TamXo2kYU0n1sPnGnNBQq1bTNTaY9OxSth8pl2RHCOEzWnSyExQURPfu3WvtCwgIICIiwrP/xhtv5N577yU8PJzg4GDuuOMOhgwZwuDBg70RsmgFlh+bX+fiLlGQc8DL0TSMpbK6X82EWx+hc89+9Sp7ugkNu7etTnYOHjVjsjgINvg1SrxCCOFNLTrZORuvvfYaarWaqVOn1ppUUIi6VNmcrN1/FKhOdta18u5aEXHtGnVCw1CjjsRwI9klZrYfKWfYsYVCW4Ka9bwaQtbzEuL81uqSnd9++63Wa39/f95++23efvtt7wQkWpXV+4qxu9wkhhtJaRPIOm8H1AL1aBtCdomZnbkmBiWHo20BMyofv55XQ8h6XkKc31pdsiPEuagZcn5xlyhUZ7HC9/mofZsAgvy1VFid7C6ooHtciLdDOmk9r/qQ9byEEJLsiPOG262QdizZGZUa5eVoWi61SkXv+FBW7StmS3YZ3WKDW0xiWLOelxBC1If366eFaCY7csspqrARoNMwMDnc2+G0aN3igvHTqDhaZSen1OLtcIQQ4pxIsiPOG2kZ1bU6F3Rsg16r8XI0LZveT0PX2OpJBrfklHk3GCGEOEeS7Ijzhqe/jjRhnZVeCaEAZBVXcbTS5t1ghBDiHEiyI84LBSYr24+Uo1LBRZ0l2TkbYUYdKW0CAEg/1DonXxRCCJBkR5wnVhyr1ekVH0qbIFkq5GwNSKru27S7oAKTpXUvrSGEOH9JsiPOC55RWF2kVqc+ooP9SQg3oCiQni21O0KI1kmSHeHzrA4Xf2RWz7wr/XXqb0C76tqdnbkmrC4vByOEEA0gyY7weWsOHMXicBET7O8ZYSTOXnyYgZhgf1xuhT0mGcUmhGh9JNkRPm95xv9GYbWUyfFaE5VKxeD21bU7ByrUaIJaznpZQghxNiTZET5NURTPkHPpr9NwieFG2oYacKMiZOg13g5HCCHqRZId4dP2FFRwpMyCXqtmaIrUSDSUSqViSEoEAIE9xpBf6fRyREIIcfYk2RE+7dddBQAMTYnAoJP+JueibaiBaH83Ko2WT7ZXeDscIYQ4a5LsCJ/287Fk55JuMV6OxDf0CHWhuF38mWNlw8ESb4cjhBBnRZId4bNyyyxsO1w9a/Lo1Ghvh+MTQnQKldt+BuCp73fhditejkgIIc5Mkh3hs345VqvTLzFMZk1uRGWrPsHop2L7kXK+2nTY2+EIIcQZab0dgBBN5edd+QBc0k1qdRqT21zOlamB/GdbBS8s2c2Y1GjCAnRnVTY7O5vi4uJ63zMjI6PeZYQQooYkO8InlZsdrD1Q3adkTFfpr9PYJnQMYF1B9Wi3Z37M4JWre52xTHZ2NqmpXTCbLQ2+b2VlZYPLCiHOX5LsCJ+0fE8BLrdCp+hAkiMDvB2Oz/HTqJg7tQdT3/2T/246zBV92jK84+mH9hcXF2M2W/jk4atJTWxTr/v9tH4vcz74BavVei5hCyHOU5LsCJ/0887q/jpjm2EUVoXJRF5eXr3KlJWVNU0wzahvYhh/G9yOj9YcYvbibSy5awSB+jP/SklNbEPfTm3rda+M7KKGhimEEJLsCN9jdbj4bU/1l+MlTdiEZXVUr4q5ceNGNu8+UK+y9sIsAMxmc6PH1ZzuH9eFXzMKySmx8MR3O3n5qjM3ZwkhRHOTZEf4nD8yi7E4XMSF+NO9bdMt/Gl3Vic7vVKiGDqgT73Krv3TwapMsNntTRFaswnUa3ntmt5Me28NX6UfZkSnNkzsFeftsIQQohZJdoTP+d8orJhmWfgzwN+P2Ij6JVXBhrMbvdQaDEwOZ9ZFHXhj+T4e+Xo7veJDaBch/aSEEC2HzLMjfIrLrfDrsVXOL+kqQ86by52jOtI3MZQKm5Ob/7ORSpusnSWEaDkk2RE+ZePBEkqq7IQY/BiQHO7tcM4bWo2ad6b3IypIz96CSu75YovMriyEaDGkGUv4lJq1sEZ1icJP07JyeY1iJ8BVgdFdQQ9jMdZkDd01B4ms2otVG0KVLhJF1XyLlZaWldV7FNnpJgSMCfFn/l/7Me2fa/llVwHPL93N7PFdmqUpUQghTkeSHeEzFEWp1V/Hy8EQ5iwkybabtrYDRDpyCXUdRUV1bce17YC/BQCfwJZPAHCptJj0cRQFdCIvqDv5Qd0pCOiMS+PfqKFZLNWT+q1YvpyV67fUq6yrsnqixlMlSX0Tw5g7pQf3LdrKeysPEGLw4/aLOpxTvEIIca4k2RE+Y9vhcnJKLBj8NIzodPoJ7ppKoKuMVPMGupo3EO48eW4YJ1qqNEGUVTkpLy8jPDKKMH8Ff2c5GsVJmDWbMGs2nY7+CoBLpaEwIJWssKEcCB9BUUAnOMeaEpvNBkD/zm0Z0Kd7vcru3pPJ91tPP0/Q1H7xlFTZefanDF5atocAnYbrhyWfS8hCCHFOJNkRPuOHbbkAXJwahVHXvB/tCEceAyrS6GzZjBo3AE405Orbc0jfmUK/eIr84rCoA0GlIu3X5fz45adcfs9tXDRiEirFRaC9iDDLIaIrM4ip2EFsxXYCHCXEVu4gtnIHQ3Peo0IXRVbYcGxqA2pVw5qiTBUVAAQZ9fUeRVYQeHa1TDePaE+5xcFbK/bxxPe7qLQ5GRIifXiEEN4hyY7wCW63wo/bqr/0L+8Z22z3jQ1U8X/R6xlTuMiz77AuhZ3GgWQaeuJQn11yoKg0VOhjqNDHkB066NhOhWBbHgnlG2hfsop2ZesIshfSs+Breurgb/cE8nXBv1n02W9srQgBzq7Gp2ZCQ6ezaUdM3XdJJxQU3l6xn5d/3suEjkZQtax+VEKI84MkO8InbM4pI7fcSoBOw8jOUU1+P5Xi5pqog8y4I5BA3SEAMv17sj5oNIW6hEa6iQqTfxw7/SexM3oSGreNhPKNpBz9naS8JcQFWZkVlMesDnkUqiLZrOnBZnVPStSnH4W2fHkV6ZngcrsaJ85Thq/i/rFdCA/Q8/QPu/gx00zUVU9gcpU06X2FEOJEkuwIn1DThDWmazT+fk07oinEWcTY0s9oG58FqNhlCWdL4gwKdIlNel+XWs/BsGEcDBvG71tjca94mfsnpjIkIJcopZixzhWMZQVHdMnsNA5ir6FXnTVLRv/mndDwxuHJRAfrue+LLZDcl/87YuHt6FwGhVc1axxCiPOXJDui1XO7FX7aXt2EdVnPpl2qoJN5M2PKPken2Klyabjrx0pyUi9mXIemTXRO5ETDj3udUHwBYzt0o4NlG10s6STYMmlrz6KtPYuLyr8m078nOwMGcliX4tUmpMt6xuE4msOsz7ZSHN6WazakcH1iMQ90zMOolb48QoimJcmOaPU2HCyhwGQjyF/LBU00CkutuBhR/i19qlYBkKPrwL2b4vhq8w9ckurdeWTsan92BQxkV8BAAlzlpJo30s28nnBnIV0tG+lq2Ui5JpxdxgHsMg7wWpztQvzI++hubpjzKr9WJrAgO5IlBSHc3zGfKXGlqGU6HiFEE5FkR7R6PxzrmDy2Wwx6beM3YencFi4rWUA7214A1gWOZk3weArsPzf6vc5VlSaEjUGj2Bh4MTGOQ3SrWk9ny2ZCXCUMqVjGkIpl9OkYxryefpSpmn9JB8Vu4R9tdvDXzg4e3dWWHIue/9uRwL8ORvKP9oVMiClHI0mPEKKRSbIjWjWny82SHTVNWI0/CivQWcoVR98n0pmHQ6VjSdh17Df0aPT7NDqVinxdEvm6JH4LnUwHyw66mdeRaMukb1Ap/7nCgNn9PQdKc9llHMARXftznr+nPi6MrOSXYXv5KDuStw5EsbvSwJ3b2vFypo1r4kuYGldKjL+sryWEaByS7IhWbV1WCcWVdsKMfgzr0LhNWCHOIq4sfpdgVylV6iC+ibi58UZaNSOXSsceY1/2GPsS5CxF2byQkfo9dIyA7uZ1dDevo0wTyU7jADKMA6jQhjVLXP4ahVuTi5gWX8JH2RF8cCiSbIuelzJjeSUzhgsjK7gmvoSL21Q0SzxCCN8lyY5o1WpGYY3rHtOoa2FFOPKYWvwuAe4KSrRt+DritmZLAppShTaMxfkp3PL9Ju65YRI3dDbTybKFUFcxwyqWMLRiKdn6juw0DmS/fw+c6qYfuRXi5+LOlEJualfETwWhfHkkjPWlgawoDmZFcTBBWhcd3AYM7Q8ga4sKIRpCkh3RajlcbpbsqF4LqzFHYYU78rmy+G2M7iqKtHF8Hfl3zJqgRrt+S7HTGskvYUNZEXIFHa3b6Fq1nkT7PtrZ9tLOthebyp89hj7sMg4kT9euyZu5jFqFK9uWcmXbUg5U6Vh0JJz/5oZRaPNjMx2IuuoJ1ljcFO/Kp2NUEInhRjTSq1kIcRYk2RGt1up9xZSZHUQG6hiUfPqJ9M5WqKOQK4vfweiuosAvnv9G/h2bOqBRrt1SOdV6Mo41YQU7j9L12NpeIa4SeprX0NO8hhJtFLuMAzikrt/yEg3VPsDOg53yub9jPhvLjLy6wckfFVEQGE5GXgUZeRXotWratwmQxEcIcUaS7IhW67st1U1Y47vHknvkMMXFxfW+RlZWlue/A52lXFn8DgHuCgr94s6LROdEJm0Ea4PHsTboEuLtB+hWtY6O1m2EOwsZbvqRoaEqxv7FiLFyK7ingbr+o98ysk9eIPV0tED3wr188WEaE+d8gNImhX2FlZjtrpMSny4xwSSEGVA1Y2drIUTLJ8mOaJWqbE5PE9bQOC1dUlOxmM0Nvp7KXsWUo/8kyF3OUW00X0fcdt4lOrWo1BzWd+CwvgPL3VPpZNlKV/N64u0HGN9RC3n/hDeWwICboM9fwXjmmrWaBUuve+7LBoels5UxtHMUF3ZqQ26ZhczCypMSn/AAHb3jQ+kSG9So/biEEK2XJDuiVVq6Ix+Lw0VyZABRmiosZjPTH3yJ6MSUel1nfdoPbPj2A+4PSyPCWUiFOoSvI27FoglsoshbH4fan50Bg9gZMIjcHX8SuesT7r4gHH1ZNvzyGKx4DnpeA8PugohT//zLysoAmHD1dXTu2L5eMaxfv4E/flmCzW4HQK1SER9mJD7M6El89hRUsCe/gpIqO8v3FPLn/mK6tQ0hpmmXABNCtAKS7IhW6evNhwGY0qctKlX10OToxBTiO3ar13Uyt23g4ysMdPMvxKbyZ3HkLVT6wKirplLoDuHlX220+9uzTOumhfX/hPztsOkj2PQf6DoJLrgXYnud8hoRUVHEt2tXr/vuydx/ymPHJz7DO0SyM9fE1pwyTFYn6YdK0ar8CBl2LRaHu173FEL4DqnjFa1ObpmFP/cfBWByn7YNv5CicFPQKq7q5odDUfNdxA0c9WvatbV8hUutg75/hVtXwcwl0GkcoMCub+CfI+DjKZCzvtnj0ms19E0MY8bQJC7rGUt0sB6noiJ0+HT+8VMRH685iMMlSY8Q5xtJdkSr882WIygKDEoOJyHc2ODr9Mn9jIkB2wCYVzyUw/qOjRXi+UOlgnZD4S9fwN9XQ4+rQaWB/Wnw7zGwcBoU7Gz2sNQqFSltArmmfwKDIh04SnIpt7mZ8+1Oxr62knUHjjZ7TEII75FkR7QqiqLwVfqxJqy+Da/VSSjbwIiDrwPwfz9bWWVOaozwzm8x3WHq+3BHenWnZZUa9i6Bd4fBf28m0FH/0XLnSqVSEW9UyP33bdzcN5jIQB0Hiqu45r21PLJ4OxVWR7PHJIRofpLsiFZlw8FSDhRVYdRpmNDAiQSDrHlM2DMbNW7SLF14ZY29kaM8z4Unw6S34Pb10HUyoMD2L7ns0FM8N0qPHi8kGG4X4zsEkHbfSK4dWL3kx6frsrnktZUs313Q/PEIIZqVJDuiVfl8QzYAl/eMI1Bf//71GpeVy3c/gMFZTkFAF94uv6ixQxQ1IjvC1R/BLb9B+4vQKE5mD9fzQtgiOpvTQWn+tR9CDH7MndKThTcPol2EkbxyKzcs2Mj9i7ZitsvCo0L4KhmNJVqNcouDn7ZXz9VyzcAGLMipKIw68DzRVbuxaEP4vsuL2Pf/0shR1k+FyeSZf6Y+TBWtaHHMuD7w18X89t5DJOx8h5RwM5eWfkKvqtWkhV5Zr07hDfl51TXZ5NCUSJbeNYLXft3Lv1YdYFH6YdKzS3nr2r50jWueWaKFEM2nRSc7c+fO5euvv2b37t0YDAaGDh3KCy+8QOfOnT3nWK1W7rvvPj7//HNsNhtjx47lnXfeITo62ouRi6bw3dZcrA43naID6ZMQWu/yvfIX0a3wR9yo+bHzc1T4xzZ+kGfJ6qie/GXjxo1s3n2g3uXthdUzPzudraQ2QqXiSGBPxr1Tycf3XszkgG20tWcxvfAVNgSNZn3QGFyqU/86Opefl6uyBOCkJMmg0/Dwpalc1DmKu7/YzIGiKia/s5o5E1K5bnA7mYVZCB/SopOd33//ndtvv50BAwbgdDp5+OGHueSSS9i1axcBAdWz295zzz38+OOPLFq0iJCQEGbNmsWUKVNYvXq1l6MXje2LY01Y1wxIrPcXUdvyzVyY9SoAfyTNIid0YKPHVx92Z/WXd6+UKIYO6FPv8suXV5GeCS5365oxz+aCHyy9KWs/gYvKvqaDdTuDK36mo2Urv4ROI0+fVGe5c/l57d6Tyfdb/zep4YmGpESw5K4R/N+irSzfXcicb3fyx75iXrqqF8H+fvW6lxCiZWrRyc7SpUtrvV6wYAFRUVGkp6czYsQIysvL+fe//83ChQu5+OKLAfjwww9JTU1l7dq1DB482BthiyawJaeMHUdM6DRqrqjn3DoBtkIm7HkIjeJiT+QY0uOua6Io6y/A34/YiPo3mxj9dU0QTfOp1ITyffhMOlq3clHZ10Q4C7im+A02B1zA6uAJONV1P19Dfl4Fgf5nPCc8QMe/Z/Tng9UHeX5JBst2FpBZuJr3/tqfDlEym7YQrV2LTnZOVF5eDkB4ePU6POnp6TgcDkaPHu05p0uXLiQmJrJmzZpTJjs2mw2bzeZ5bTKZmizm7OzsBi1QCRAZGUliYmIjR9Q6/WfNQQAu6xlLeMDZf9Fr3HYu2/MQAY4Siowd+LnDnOq5YcQ5ycrKYtOmTfUuU4tKRaahN9n6jlxY/h3dzOvpW7WSdrbdLA27jkJdA/plnQOVSsWNw5Pp1y6Mv3+cXt2s9fZqXr26F5d0i2nWWIQQjavVJDtut5u7776bYcOG0b17dwDy8/PR6XSEhobWOjc6Opr8/PxTXmvu3Lk8+eSTTRkuUJ3opKZ2wWy2NKi80WggI2P3eZ/wlFTZ+WFbdX+Lvw6p3zIDIw+8TFzFdqyaIL7v8hJOjaEpQjxvVFZU/2EwZ84c5syZ06BrmG21+xnZ1AH8HHYtewx9uKT0cyKchUwrmsfaoLFsCBqFoqr/yurnondCKN/fMZzbF25ifVYJt3yczp2jOnL3qI6o1ZIoC9EatZpk5/bbb2fHjh388ccf53yt2bNnc++993pem0wmEhIa/6/I4uJizGYLnzx8NamJbepVNiO7iOue+5Li4uLzPtn5YkMOdqebHm1D6F2Pjsnd87+hZ8FiFFQs6fw05Yb4pgvyPGGzVCfuI/5yJ/2HjaxX2fVpP/DH1x9gc9TdqfqQfxf+E30/o8sW0cmylWEVS0i2ZbA0bPq5hl1vbYL0fHrTIJ79MYMFfx7kjbRMdh4p59VrehNikH48QrQ2rSLZmTVrFj/88AMrV64kPv5/X1gxMTHY7XbKyspq1e4UFBQQE3Pqame9Xo9er2/KkGtJTWxD307nsIbTeczlVvhk7SGgulbnbDsmx1Ts4KIDLwLwZ+LfORg2rMliPB+FRMfXe9HVPdvSz3iOTR3Aj2Ez2O+fzsVl/yXOfpDphS9zKLQLqxoabAP5adQ8MbEbcf5OXv79MGm7Cxn3ShoPDQsjIeTsEh5pihaiZWjRyY6iKNxxxx0sXryY3377jeTk5FrH+/Xrh5+fH2lpaUydOhWAPXv2kJ2dzZAhQ7wRsmhkaRkFHCmzEGr0Y2Kvs5uPxWg/ymW7H0SrONgXPpL18dc3bZCicalU7Db254guhfGlH9PWnsWT7bcSO8GfH1XNO/osOzubu68YiisoljZXPEIeUdzxXTbFP76KJXPtGcsbjEZ2Z2RIwiOEl7XoZOf2229n4cKFfPvttwQFBXn64YSEhGAwGAgJCeHGG2/k3nvvJTw8nODgYO644w6GDBkiI7F8xPurqudUuXZgIv5+Z+67oXY7mbBnNkH2Qo4akljW8fHqNZpEq1OhDWNR5O0MqVjGANMv/L2/jktsaaxydKLEr3nm0SouLsZiNjP9jr8T2jaUdcVuijASNeVRUoNdpIa4TtnfvSB7P5++cL80RQvRArToZOfdd98FYOTIkbX2f/jhh1x//fUAvPbaa6jVaqZOnVprUkHR+m3KLmXDwVL8NCquH5p0VmVGHJxHvGkzNk0A33d5CbtWhg23ZopKw5/Bl/LFphIejN1A+8ByEopeZXnIVHYFNN9cSdGJKcR37EayW2HVvmK25JSRYdJg1QUztls0em3zdqIWQtRPi/6TV1GUOreaRAfA39+ft99+m5KSEqqqqvj6669P219HtB7/OlarM6l3W6KDzzxXSj/LavrkfQHA0o5PUmpMasrwRDPaWBFJr/lVbDZH4afYGVv2GWNLPsXPbTtz4UakVqu4sFMbLukajUatIqu4ii825FBSJYvJCtGStehkR5y/so+aWbqjutny5gvan/H8PjFqrqpYAMDa+Bs5EHFhU4YnvKCgSuHhIyNYHXwpblR0tWzkL0WvEuHIbfZYUmODuapfPIF6LaVmB19syOFAUWWzxyGEODuS7IgW6b1V+3ErcGGnNnSOCTrtuRpbOV9fY8QPBwfChrEm8ZZmilI0Nzcq1geN4avI26lQhxDuLOQvhfPoVrW22VdRjw7259qBCbQNNWB3ufl+Wx7rDhxF8cJq7kKI05NkR7Q4eeUWvtxwGIDbRqac/mSXk+T0p0gKVVOkiWZpp6elQ/J54Ig+hU+j/o8sfRe0OLik7AvGlTZ/s5ZRp+WKPm3pFR8CwNqsEn7cnofN2brWLBPC18m3gmhx5v+2H7vLzaDkcAa3jzj9yWlPEly8iUq7woKQO7BpT18LJHyHRRPINxE3syr4MtyoSbWk85eiV4ls5mYtjVrFyM5RjE6NQqNSsb+ouh+PyS6zLQvRUrTo0Vji/JKdnc3enAIWrisE4NJETrv+UtiR5SSnvwHADd9aSLg9Hpkj+TyjUrMxaBS5umQuLfkP4c5Cri2cx4rQKewwDmrWddC6xYUQEaDnx+15lJodLLdoCeg+qtnuL4Q4NUl2RIuQnZ1Nl9RU/IdcR3D/iVgP7+T6Sx885fl9Y9X8MTMA/FS8uNrGol1ObqmUDqLnq1x9ez6J+j/GlS4k2ZbBmLIviLftIy30qmaNIyakuh/Psp0FZJeYiZxwD2+uL+Od7i4MOhmeLoS3SLIjWoTi4mLs2kCi+1+GAozu04nooV/XeW6Qq4y7S5/E4C5ll64nK9v2Ad7EarU2a8yiZbFqAvkm4ib6V65gmOknUi3pRDtyyNc071IhRp2Wyb3j+DV9NztLVaw4aGHS23/w9l/60jFamlmF8AZJdkSLEXrBdSioiQ8z0LdHhzrXwdK4bVy1/VZC3aUcNSSzoufrhJqWA1BaVkZeXl697mmqqGiU2EULcXyzVunHhDsLeSzkO/L7+DXraC2VSkVqiJsV85+g88zn2VtQyWVv/sGjE1K5bvDZr/EmhGgckuyIFuFAqYPA7hcDMLxDZN1fBorCmH3PElu5E6s2mG9TX8GuDcRybCXuFcuXs3L9lnrd116YBYDTWfdK3KJ1ytW355M29zG2dCHtbRn8a6KBg/kfgmUCGMKaLQ5r9jZeGRPJgt1uVmUWM+fbnazYU8SLV/YkMrD5FiMW4nwnyY7wOkVR+HibCYAEo+uUsyX3P/IxqUVLcKPhh87PU25IAMBmqx5u3L9zWwb06V6vey9fXkV6JrjcMlTY11g1gXwbcROxmV9wpWEdSZUb4d1hcMV8SB7RbHGEGTR8NLM/H/55kBeW7Gb57kLGzVvJS1f24qIuUc0WhxDnM0l2hNf9squArQV2FJeDrqF1NzV0KvqZCw69CcBv7e8lJ3TASecEGfXERgTX695Gf139Axath0rNEmtPXvtsBb/clkSw6Qh8NBGG3A6jHgNt09euZGRkANDHCM+PCue1tWXkmOzMXLCBUckGru8VTIDu5FlAIiMjZQFRIRqJJDvCqyx2F09+vwsA0/rFBCZPPumc+LKNjM18AoDNsVezNaZ5R9iI1m/9ERdLEmdzTeg2SF8Aa96C/SvginchtleT3NNUUgTAddddV2u/Sqsj9MIZBPefRFqWhWXbcji69C2sBzbWOs9gNLI7I0MSHiEagSQ7wqveWpHJkTILkUYN2Wu+gGsm1zoeWZXJxN3/h1ZxsDdiFL8n39usc6cI3+FU+8Plr0OncfDtLCjcCe9dBMPuhAsfAr8zLzZbH5bK6qbZCbc+Quee/U46Xmx1sLFES1VQJNFXPUG7ABc9w1zo1FCQvZ9PX7if4uJiSXaEaASS7Aiv2VdYyXsrq1c2v7FPMOkOG8XFxWgCq0dUhTqLmHjkfvSuKg74d2VB0G048wtPuo6MqGp+FSZTqxv5lpWVdWySymi0F7xHwvY3CMv9Df54DevmrzjU+36qInrUKlPTBHUuIuLaEd+x20n744FuLjdr9h9lc04Zh6o0FNp1jOgUSVTCOd9WCHEcSXaEVzhdbu5btBWHS+HiLlEkqKqTmK+//hpNYDihWjtL+/9JSGAluyqDGP9bPOXOBXVeS0ZUNR+ro7oj98aNG9m8+0C9ynrrfaqsqK5hmTNnDnPmzKl1bFJnLe9O8CeWHDqvvpN3N9p5dLmNEkvtvmOVTTRhpZ9GzYhObegQFcivGQWUmh0s21lAG70WbVhck9xTiPORJDvCK/658gBbc8oI8tfy7BXdWf5D9QSCF/VpT59O8dxs/4h4pZIyglkccRPXjAs55bVkRFXzsR9b4LJXShRDB/SpV1lvvU+2Y1MTjPjLnfQfNvKk42+7q7i88nMGWVdxW38dfxsQxpKAqaw1jGTXhlUs+ej1Jp+wMi7UwPRB7UjPLmV9VglFNjVxN7zN5zsqSJXZl4U4Z5LsiGa3K9fEvF/3AvDUpG7Ehhg8x+KD4B/uT4hW8jCrA/gu8u8Y/WIxnuZ6MqKq+QX4+7W6kW8h0fF1NicB/MlAsss2clHWy0Sa93Nl5X8Yoazhn7FDWdJM8WnUKgYmhdM5Ooglmw5QYPXjy12V/PHKbzw4vgsTe8XJZIRCNJCsei6aVYXVwayFm3C4FMZ2i2Zy77aeY2H+8EDwEqIdhzGrA/kq8naO+sV6MVpxPjkc2p9Pen/C8vb3Y9UEEVWVyRz9R3xzjYEEpflWUg8x+DGsjZOib+YSadSQW27lrs+3cOX8NWzNKWu2OITwJZLsiGajKAoPfLWNA8VVxIb489wVPTx/qepcVfz6twCStEePJTr/kERHNDtFpWVr7NV82O9rtkVPwa2omNTFj2eUV7l0z8OEmQ82SxwqFZj3rObNcW24b0wnDH4a0g+VMunt1dz9+WZySszNEocQvkKasVq4ho4GOZcJybKzsykuLm70+/5rVRZLduTjp1Hx9vS+RNRMl19RwKgjrxMeq6Hc7c+3UZLoCO+y+oWS1mE2H2dH03fvG1zbw4/Oxb/QsTiNPW0uYVPcdAoDuzR5HHqtijtGdeSq/gm8sHQ3izcf4Zstufy4PY/pg9ox6+IOsuyEEGdBkp0WKq+kAhUnT0h2toxGAxkZu+ud8GRnZ9MlNRWLuWF/OZ5qIrSlO/J5bkl14jbnsq70TTy2PlHxPvjkCsJthymodPO6cwJtEiTRES1DnhLJX762sLPng1wflk6Hkt9JLVpKatFScoL7sSnuWg6EXwCqpq0kjwnx57VrejNzWBIvLdvDqsxiFvx5kC835nDT8GRuHN6eEKNfk8YgRGsmyU4LVVZpRQHe+sclDOnZsV5lM7KLuO65Lxs0IVlxcTEWs5npD75EdGJKvcqeaiK09EMl3PX5ZhQFrh2YyF8Ht6s+cHA1fHEdWEqo8GvDsA8OMGx6GG3qdVchml6OKo7vU68lqjKDvkc+pdPRX0kwpZNgSqfUP4Ed0ZPJaDOeKn3Tfnp7xofy8Y2DWL2vmBeW7mbb4XLeWL6PD1cf5PphSdwwLJmwAOmwL8SJJNlpBhnZRfUuk5VfCkCHuDD6dmp7hrMbX3RiyilHrtTHjiPl3PjRRmxON6O6RPH0pG7V/XQ2/AuWPAhuJ8T1ZZn+GvaX3sawRohdiKZSGJjK0s7P8IftDnrnfUmP/MWEWXO44NCbDDv0NofCBrMr6jL2hzftQqPDOkTy7e3DWLIjn9d/zWRPQQVvLt/HB39k8behSdw4PFmat4Q4jiQ7TahmhtnrnvuywdcoLKtqrHCa3ZacMv7273WYrE56J4Ty5l/6oHWa4fsHYMun1Sd1nwoT38K2aLF3gxWiHir10fyRdAfrEm6kc9HPdC36kbamLSSX/kly6Z/YNAEM8GuHo5sWf6Vp5uhRqVRc2iOWcd1i+HlXPq+n7SMjz8S7v+3ngz+ymNI3npsuSCalTWCT3F+I1kSSnSZUVlYGwISrr6Nzx/b1KrtxWwYrv19EudnWBJE1vdX7irn143QqbU76tQtjwcwBGI/uhK9ugKP7qvs4jHoMht0ta12JVsuhMbIjZjI7YiYTasmma+GPdC38gSB7IUM0uxhypRGH8jg5u5ZwKHQIh0IHUWpo16ifebVaxbjusYztFsOvGYW8tTyTrYfL+Wx9Np+tz2Z0ajQ3XZDMoORwmadHnLck2WkGEVFRxLdrV68ymUcaNhqqJViyr4oPtqzH5VYYlBzOB9N7ELD6eVg9r7rZKigOpr4PScO9HaoQZ6W0rOyMa4Hl4UeGbjJft51Igm0vMXs/p6dtA50ioH3patqXrgagQhdFdugAckIGkBvUi3L/tqdMfuo7GjMCeGyIgVxHBN/uqeLXjALP1jk6iOsGJzK5T1uC/E/dmbmpRmMK4U2S7IhG43BDxPi7eH9T9VpEV/SOY26PfPw/HAlHM6tPSp1YvfK0Mdx7gQpxlizHlppYsXw5K9dvqVdZe6ENS2YVdz/0MFM6QbuydcSZthJkL6Rb4Y90K/wRgCq/cHKDepIX3JPcoJ4UBqZ61uI619GYsy/twr//yOLrTYfZU1DBnG93MnfJbib3acu1AxLp3ja4Vm1PU43GFMLbJNlpQVSKixBXCaHOIsYGZtJ7hI7hAYdg71JQ3KDWgtoPNMf+1RnBP6R60wWB2nvr5+SWWUjL9yOw5xhUwP2DjdxW9jiqRb9XnxAYDZe+DF0nei1GIerLZqtuRu7fuS0D+nSvV9m1fzpYlQn7baFsjJ/ExvgZaFxW2lZsJbFsPW1Nm4mq3E2Ao4SOJb/RseQ3AJwqP7KUOLqM1dOuQzc6pfbArgk466av40dj9u2byHNX9ODBcV34etNhPll7iP1FVSxcl83Cddl0ig5kSt94JvduS0yIf5OMxhSiJZBkx4sMrkoSbJm0tR8gxp5NpOMIWo4tkhgBXOQP5ENe/llcTQX6IAhoQ5zNwFVdtegrs8Hdq0mToCqbk9X7i8nIqwBU+JkLeS/xFy7aUv1XKxodDP4HXHBvdVImhJdUmExnbIo6kamiAoAgo77ea4EFG04eAu7S+JMdOojs0EEAaNw2oit3E2vaRlzFVuIqtmF0lNKRQ9wzWA/sg6J94BcAwXHVW1AcBMWA9vSjrU5sAutlgJ4jg9hZpGfZfjPrj1jZW1DJ80t28+LS3fSI0pGsLUNtCG600ZhCtBSS7DSzUGcRHS1b6WDZTowj+6TjDpUfZZo2HDCp2JqRRb/UdvRLTalOWNxOcDnA7aj+114JVhNYy0Fxgc0ENhMxwJdXGWH5DFhpgKhUiOkOUd0g+th2js1IFruLTdmlbD1chsOlAHCFZjVPhn1AcKWluuap97VwwX0QlnRO9xLiXFgd1X9AbNy4kc27D9SrrL0wCwCn09nocQG41Hpyg3uRG9yLdP4KikKoNQf75q9wrPuIa/tHEqGpBEdVdVNwTXMwKgiIrE58apIgYwSo1OSVVCdoZ2oCU+kDCOgynIBuF+Of0I2tBXa2YiR+1sesyHXTy1BG+zaBBOrla0K0fvIpbgZ+OEk1b6BH1Rra2rNqHSvSxpKj70ieLokCXQLlmnBQqVmxP53vf8jgzcRE+p2pI6+igL0KLKVQVUjxkSz2Z+5hQDsjaqcFcjdVb8cLjIHortWJT0QHCG4LQbFo7OWnvIef20JlSQHbjpSTXuKPTamuMeql2s8Tfgvoo95PQaUbc++/EHP5YxDS/PMDCXEiu7M62emVEsXQAX3qVXb58irSM8HldjVFaCdTqSgzJLJF1Y//LPknYX2GMn1EN6gsAFMuVORW/2szQVVR9Za/tbqsRgdBsbQzaxiboiFozL0k9hhyVretdNg5bFazv9iM1S+YEqeGFXuKWLGniMhAHUkRASRFBBAb4o9aLSO6ROsjyU4TCrUd4Y3x/twQtpCAUjsAbtRk6zuyz9CT/f7dMWvqVzVeJ5UK9IHVW2gC2VUxDP73JtI3rqBvu1Ao2AH5O6BwFxTshLJDUJlfve1fXutSvQDLI0G4imbhLjOiKJDjDmeFPZUlzv7sVJKBAAB6qA5wp/Zrhuv2cTB8GB/aL+XWp59l7YZbiZFER7QwAf5+9W6KMvqf+2zEDWk+Kz02bUVWfimb9hcCKqAtaNtCOGhdZgIcxQTYiwlwFGN0HEXjskPZIbprYOl1Abh4n0Lzag6H9CUnpD+5wb1waIynvGcXICztO76Yfw8X3PEqZkM0+SYrxZV2iivtbDxUik6jpm2YgfhQA23DDLQJ1Dda8iOjwERTkmSnqTgsXHL4FfwG6gA75ZpwdgQMZqdxIFWaZuq7otJAZMfqrdsV/9tvq4DCjOrEpyb5MeWB6QhYSijRRLDG2Y0/bV35092dPCI8RbU4Ganbw5jgHKIiI8gLvpP3AjrjVms5nLkTh7t5Hk2Ilu5cms8cJUcAmPPBL8z54Jcznq9WQbc2agbFaxgar+HCJC3twyC2cgexlTsYcOQ/uFQaCgK7cjikHznB/cgN7oVTYzjpWs7yAhL8qug9IAGz3Ul2iZmDxWYOlVRhdbjJKq4iq7h6slOdRk1MiD8xwf5ozCo0AWH1es4aMgpMNDVJdpqKn4EDQYPYvuZXsuIvR9tlTJMvFnjW9EGQMJCKNn3IibOQWVjBrjwTGXkVbDtUTJlNqXW6RgWJwWo6RAeTHBWGQZ9KLpDrneiFaBXOufksAwaPm8TQfr3qVfaD9Ru44Y0l3HDvw1zVN5KE8o3El28ixJZLXMV24iq2M5AFuFRa8oJ6kB06kOyQgeQHdT3pWkadli4xwXSJCcatKBRV2DhSauFwmYUjZRbsTjfZJWayS8yAH/GzPubm7wvov3MjvRJC6RobTOeYIGKC/U87oaGMAhNNTZKdJrSxzdVct+h7/jYrnt5eSHQsDje7800cLrFwuNTM4VILh0st5Bz773KLo85yittFuL+KlNgIEsKNxIb446dpIYmaEK3MuTSfBYdH1ntC0j2Z+wE4UK5muasvBPaFQAhzFNDeuoMUy3ZSrNsJcxYRb9pMvGkzQ/knVpWBrcQQMUiH05RJXm7HOoe7x+ogNkpFvzYGyq1ujlrclFhdFJZbqHJrOGqBZTsLWLazwFMm2F9L55ggOkUHef7tGBVIxAnrd8koMNFUJNlpSk08NbvNrSLHrCPboiPHrOOwVcdhix97S9sRf8dQpi8uAApOe43wAB1JEUa6xgXTNTYEtSmXv0wYyZXzPiO+Q2STxi+EaHxnbj4zAgNJNpgZGV7MheHFjAgrJlxnYZB/FoPG+QP/JH/3An4vifRsR2wnN3kdz1VZQlXG77z6wSKMCV3YdricPfkVHCiuwmR1suFgKRsOltYqE2b0o0NUIKFqK0H9J5JvURFsdRCk18rSFqJRSbLTwilAgVVLRoU/B816ssx6sqr0ZJl1HLHocFP3L4SafoihRj8SwozEhxmObcZa/wacMKx006ZiFEfrXI9LCFH/5rM/gTWKmzglH9XeX+ji2suFyTpi9DauiT3CNbHV/YeKVBFkqtuzT92e/epkLKrayc/uPZl8v3UpbShj+oj/NUXZnC4OFFWxt6CCPfnHtoIKDpdaKDU7PAlQ+KhbWF0Eq4sO4qdREWbUER5w3GbUEWLwk9FgokEk2WlhLC4VuVYd+QGdiLryCd4NSOWV3wNOeX6gxkWi0U6CwU68ofpfW2k+dzz3AQ/deQud2x9XBV5WveVnQT6wsY7rZWVl1bFXCNHa1Lf5zE0oi9ceZNX3W5gw/WpmDI4h0ZZJonUv0Y5s2ihHaeM6ylDXBtyoKPSLJ1vfiWx9J3L1yRQE+td5Xb1WQ2psMKmxtWOx2F3sL6pkf1Elq7bu46NvlhHTbShVLhUOl0JhhY3Citp/eKlVEGbUEXYs+QkP0OGwq1Bpz33UXHOT0WfNS5IdL7O4VORYqmtpjlh0HHUce0sCQzEEghlQo9A+wEZKgI0ko432ATaSA2wkGe200TlPai37MGMrjuJDPP3YIw2Oq6CwQNrOhThPORQNh/UdOazvyJ/Bl6J3W2hr20eibS+JtkwinAXEOHKIceQwsDINJ1r2BEURN1xHhPVg9aSnmlMvNgpg0Gno3jaE7m1DSHAX8Mr1c/nbmK+JTelKucVBSZW9ejPbKT323063wtEqO0er7MddyY+Ee7/ith8L6bZtAx2iAunQJpCUqEA6RAUSYjh9HN4go8+anyQ7zUxR4KhdU90cZdaTb/VDOaEpKsLPiduUx76V33Hn8AgendQNg0Y5xRVPdqS4egbVIeMm07FD+3rFt33rNjav+hWTqaJe5YQQvsumNnDA0IMDhh4ABLjKPLU+ibZMAt3ldNPlMneUP+S8CM+/BW37QcJASBgE8QPOetZ2jVrlabo6nqIoVFidngSoJhkqrrDgcKspqHJRsLuQ5bsLa5WLCtLTOSaI1NhgusQE0SUmmA5Rgei03ht0IaPPmp8kO81AAQpsWjIr/cms9MfkrL1WVaTOQbzBTlt/B3EGO0aNwoo/t7NlyxJih47BoDl5SOjZSIiPo3fP+tXOHC0sYHOD7iaEOF9UaULJMA4gwzgAFIUwZyGarN8JzF7JuNRADA4zHFxVvR1jDUykMrw7VeHdqArrhjUwwTMdx4nreNVFpVIRbPAj2OBHEv9r2s/Zu5PXH7iBW+9/HG14W4psGgptagqtGiqcak9z2KrM/zUZqVFo4+8mxt9FQpCa3kltSAr1I8xfXa+O0efanCSjz5qPJDtNqMCqJvSC69hgGIzl8P9mLtWoFBIMdtobbSQF2AjSykx8QohWSqWi1C+aP4oS+PpLCyosdIlUMzRB49m6RGrwr8zGvzKbyOyfADDZFDbluUjPc7Ex10WHcDVVFaZ6377g8EHc5jLeffKek0PTGfGLTEDXJgm/Nkno2iShi0oG/0AKrBoKrBq2lsEPOdWdpF3mcuyFB3EUZWEvOoijMAt7cXZ1s1wdpDmp9ZBkp4nYnC7+uT+IkKHTsABalUKy0UbHQCtJRht+Mm2NEMKH2CwWAC74y530HzYSgJ3HtgB3BYmO/SQ59pHkyCTRkUWw3s7IJC0jk/73NWS2PEDxju4UBnShIDCVgsBUyv3jTzuNR3npUQD6jZlEt85nahIqQnEVYTPrqVIHsr+gkmKzi5AOfXH6BaAxhmBI6oUh6fiJHBWCtBCicxPipxCiUwj2U6jI3c9CaU5qNSTZaSJ6rYYeIXZWrttEv8RgRqa2Qac++343QgjRGoVEx9fZNFPBYLYD2wGV4iTcfJDoygyiKzMIyP2TeNdhjH5WEss3klj+v7GiVk0QhYGdKQro5NlKDMm41bW/vuLatq13sz15y9n93acMuedFLhh5OUer7BRX2o6tB2ajuMKG1emmwgkVTg2HjyuqpgtxN73Lc3+U0Dt3F+0iA0iOCKBdhJG4UAMaGSLfokiy04Qmt7Xw38XPEjXrXnTqhk3Ql1NkYtPeI/Uqk1tc/6rgEzVk8cKaYZRn0/5+IhnyLoRvONvfHbkEsIP+YOzPrsIYlr3/LA8+9gQjUgKIOpYEtanKxN9VcVIC5FT5cdTYnqKAThiN5ZgSNRhV9tPc7cy0GjXRwf5EB/9vCL2iKFTZXZ7EpyYJKjM7cCngF5HAxlwbG3Nr//7SadQkhFfPZ5YQbjg215nRs09R5A/f5ibJThM6lwlAzZXVo6FeXLSOFxeta9A1LHZnvcucy+KFrsoSAK677rp637eGw1H/mIUQ3ncuvzvshVk43bDHEkpE9Hh2Rk8EQO12EmHeT5uqvbU2f1cl0VV7iK7aQ/dguHVmAPAl5fm/UujXlqLjtgpNaIN/GatUKgL1WgL1WpIi/tcp2q0oZO7exUevPcWcF97AZYzg0NHqBVJzSizYXW72F1Wxv6iqzuv6a1XEznyTP4u0RLuLCDZoqztf+/sRbNCi12rqLCcaTpKdFspuswINWwhwxco/2bzqV2zHZlKt133PYfHCDZt38NtWePCqwVw9ql+9yv7rp428+906XC5JdoRojc7ld8faPx2sygSbvXbtjFutpSiwM0WBnf+3U1EItuV5Eh8yfyXWup+kUDUhrhJCXCV0tG73nG5VGSn0i6tOfnTVCVCJNhq3quEJhVqlIkAL1oNbGNchgL59/zdi1uVWyC2zkFNiJqfUTM6xtQlzSqv3FVbYsDoVdFHJ5Fkg73DZSdfXa9WEGPwI8tcS4kmC/Aj21+KU8SwNIslOC9eQhQADQ+vfjHSihixeGGSsXtQvvk0QfTu1rVfZuPV763W+EKJlasjvjmBD9Zw6Z9sEloeKPXQGv86szXGwauFWrr3hRq7oE0UbxxHPFuHIx18xk2jfR6J9HxyraHGi4ahfDN1j1XQarCNIs58AWyFVujbnvKahRq0iIdxIQrixzuNWh4tfVqdz9Q23Me7Wx1AHt8FkcWCyOjBZnFgcLmxOd50zSFfTET/rYx78tZjUPZtJjgygfZsAUtoEkhQZQKBevtbrIj8VIYQQXnWuTWAAZXYNh/UdOKzv4DmmUZyEO/Jp4zhClOMIbRy5tHEcQa9YiXYcYUwojBnrD3wGGz/DqgnkqDGFo8b2x7bq/zb7hTfaws7+fhraBmuxZm2ifZD7pAWX7U73scTHgcnq9CRC5ZbqZMjucqMJCCOzxEFmSe5J148O1tM+MpDkNgG0j6xOgpIjA4gPM6DVnL/DgCXZEUII4VXn0gS2fHkV6Zngcp/cbO9SaSnSxVOki2dXzU5FIdhVQhvHESr3rMavcCeDu8TQVlOGv6uSthVbaVuxtdZ1zNpQjhrbU2pIoswQT5l/Ag6nHf8m+AbVadVEBuqJDNTXefzAnp2898z9vPreR2hCYjhQVMmBour+Qker7BSYbBSYbKw5cLRWOT+NisRwI+3bBNL+WCLUvk0g7SMDCA/Q+fwq8z6T7Lz99tu89NJL5Ofn06tXL958800GDhzo7bCEEEKcpYY0gRn967kIqEqFSRuBSRtBWnExP361kcvvuY3RY8cTZjlEhPkAEeb9x/49QKj1MEZnGUbTJhJMmzyXmQg88Egw9p+vhm1dICS+egtue9y/bcE/pH7xnYFODfaC/YRWHCQ13sDAIKC9P+BPpd1NboWTIxVOciuc5Fa4yK1wklfpxO5STtlpOthfe1IS1DbUQGyIPxGBes8w+ta8eKlPJDtffPEF9957L/Pnz2fQoEHMmzePsWPHsmfPHqKiorwdnhBCiBbOpdZRHNCR4oCOtfZrXVbCLQeJMO8n1JJNqPUwodYcQqoOYlAs6KxFcLDo1BfWBUJAGwiIrP7XGEFchYu7B+voZ1lNwNFi7JoA7NoA7JoAbJpA7NoAXCpdnU1nppLqe9Vv1KsKTXAkfuHxGGOSmDHrAYqsKg4UVZFbbsFkdbIlp4wtOWUnldSoVUQH6QnzV7Hpj+XYygtxmctxWypwW0y4LCbcZhNuawUuSwUqlwONunqFeo0Kz38HBRj4M30niUnJ9Yi78fhEsvPqq69y8803M3PmTADmz5/Pjz/+yAcffMBDDz3k5eiEEEK0Vk6NP4WBXSgM7FJr/+G9O1hw/5Ws/u4/dInSQXkOlB8B0xEoP1y9WcvAXlm9lf5vLp4Y4LWx/lDxPuyu+74KKlxqHU6VDpfaD5dah0ulo9Ju4dabAzBGxqMPCOZ/M/aoPOXquhaA01pF8eFl9D1aQKDRAOEurCFqDjqCOWAPI8sRxn5HOFmOcPJcIRS6g3C51eSWW8ktB33HIdTduPY/WpwEYK3eVFaMWAk89u/+Pdsl2Wkou91Oeno6s2fP9uxTq9WMHj2aNWvW1FnGZrNhs/2vl3t5eTkAJtO5T8Z3PLPZDMC+fQew2+vqVX9qBbnVHc9yc3JYv3HTGc5uGWVzDh0E4M9dR9D/uKFeZTfurp6bNOvgIdT1nHm0Nf6szqWsN+8tZaWsL5XNyamesDVnz3bW+5/pa7y2kvwcSiwKn/y2m3bt2gHBx7bU6m/WCNAodozOcvSuCvSuKvxdVehdldhKc8nauZGOHZII0qsxYMUfGwasGKgZfq8A1mNbNQ0QAoREqEHJhcqTOyifSXyMBnfedo7/tos7tg0/4Vwnao4STL4SRqESRoESRqESShmBlCnHNgKO/RuEGzV2wI6aUoxA7RFpEyoqG/17tuZ6Z5yoUWnljhw5ogDKn3/+WWv//fffrwwcOLDOMo8//rhC9SdJNtlkk0022WRr5VtOTs5pc4VWX7PTELNnz+bee+/1vHa73ZSUlBAREdEoPdJNJhMJCQnk5OQQHFy/znatga8/H8gz+gJffz6QZ/QFvv580LTPqCgKFRUVxMXFnfa8Vp/sREZGotFoKCgoqLW/oKCAmJiYOsvo9Xr0+tpVlqGhoY0eW3BwsM9+eMH3nw/kGX2Brz8fyDP6Al9/Pmi6ZwwJCTnjOa1+hiGdTke/fv1IS0vz7HO73aSlpTFkyBAvRiaEEEKIlqDV1+wA3HvvvcyYMYP+/fszcOBA5s2bR1VVlWd0lhBCCCHOXz6R7FxzzTUUFRXx2GOPkZ+fT+/evVm6dCnR0dFeiUev1/P444+f1FTmK3z9+UCe0Rf4+vOBPKMv8PXng5bxjCpFOdN4LSGEEEKI1qvV99kRQgghhDgdSXaEEEII4dMk2RFCCCGET5NkRwghhBA+TZKdRvb222+TlJSEv78/gwYNYv369d4OqcFWrlzJ5ZdfTlxcHCqVim+++abWcUVReOyxx4iNjcVgMDB69GgyMzO9E2wDzJ07lwEDBhAUFERUVBSTJ09mz549tc6xWq3cfvvtREREEBgYyNSpU0+awLIle/fdd+nZs6dnMq8hQ4awZMkSz/HW/nwnev7551GpVNx9992efa39GZ944glUKlWtrUuX/y1K2dqfr8aRI0e47rrriIiIwGAw0KNHDzZu3Og53tp/3yQlJZ30PqpUKm6//Xag9b+PLpeLOXPmkJycjMFgICUlhaeffrrWmlVefQ/PfXUqUePzzz9XdDqd8sEHHyg7d+5Ubr75ZiU0NFQpKCjwdmgN8tNPPymPPPKI8vXXXyuAsnjx4lrHn3/+eSUkJET55ptvlK1btyoTJ05UkpOTFYvF4p2A62ns2LHKhx9+qOzYsUPZsmWLcumllyqJiYlKZWWl55y///3vSkJCgpKWlqZs3LhRGTx4sDJ06FAvRl0/3333nfLjjz8qe/fuVfbs2aM8/PDDip+fn7Jjxw5FUVr/8x1v/fr1SlJSktKzZ0/lrrvu8uxv7c/4+OOPK926dVPy8vI8W1FRked4a38+RVGUkpISpV27dsr111+vrFu3Tjlw4ICybNkyZd++fZ5zWvvvm8LCwlrv4S+//KIAyooVKxRFaf3v47PPPqtEREQoP/zwg5KVlaUsWrRICQwMVF5//XXPOd58DyXZaUQDBw5Ubr/9ds9rl8ulxMXFKXPnzvViVI3jxGTH7XYrMTExyksvveTZV1ZWpuj1euWzzz7zQoTnrrCwUAGU33//XVGU6ufx8/NTFi1a5DknIyNDAZQ1a9Z4K8xzFhYWpvzrX//yqeerqKhQOnbsqPzyyy/KhRde6El2fOEZH3/8caVXr151HvOF51MURXnwwQeV4cOHn/K4L/6+ueuuu5SUlBTF7Xb7xPs4YcIE5YYbbqi1b8qUKcr06dMVRfH+eyjNWI3EbreTnp7O6NGjPfvUajWjR49mzZo1XoysaWRlZZGfn1/reUNCQhg0aFCrfd7y8nIAwsPDAUhPT8fhcNR6xi5dupCYmNgqn9HlcvH5559TVVXFkCFDfOr5br/9diZMmFDrWcB33sPMzEzi4uJo374906dPJzs7G/Cd5/vuu+/o378/V111FVFRUfTp04f333/fc9zXft/Y7XY++eQTbrjhBlQqlU+8j0OHDiUtLY29e/cCsHXrVv744w/Gjx8PeP899IkZlFuC4uJiXC7XSbM2R0dHs3v3bi9F1XTy8/MB6nzemmOtidvt5u6772bYsGF0794dqH5GnU530iKxre0Zt2/fzpAhQ7BarQQGBrJ48WK6du3Kli1bfOL5Pv/8czZt2sSGDRtOOuYL7+GgQYNYsGABnTt3Ji8vjyeffJILLriAHTt2+MTzARw4cIB3332Xe++9l4cffpgNGzZw5513otPpmDFjhs/9vvnmm28oKyvj+uuvB3zjc/rQQw9hMpno0qULGo0Gl8vFs88+y/Tp0wHvf2dIsiME1TUDO3bs4I8//vB2KI2uc+fObNmyhfLycr766itmzJjB77//7u2wGkVOTg533XUXv/zyC/7+/t4Op0nU/GUM0LNnTwYNGkS7du348ssvMRgMXoys8bjdbvr3789zzz0HQJ8+fdixYwfz589nxowZXo6u8f373/9m/PjxxMXFeTuURvPll1/y6aefsnDhQrp168aWLVu4++67iYuLaxHvoTRjNZLIyEg0Gs1JvecLCgqIiYnxUlRNp+aZfOF5Z82axQ8//MCKFSuIj4/37I+JicFut1NWVlbr/Nb2jDqdjg4dOtCvXz/mzp1Lr169eP31133i+dLT0yksLKRv375otVq0Wi2///47b7zxBlqtlujo6Fb/jCcKDQ2lU6dO7Nu3zyfeQ4DY2Fi6du1aa19qaqqnuc6Xft8cOnSIX3/9lZtuusmzzxfex/vvv5+HHnqIadOm0aNHD/76179yzz33MHfuXMD776EkO41Ep9PRr18/0tLSPPvcbjdpaWkMGTLEi5E1jeTkZGJiYmo9r8lkYt26da3meRVFYdasWSxevJjly5eTnJxc63i/fv3w8/Or9Yx79uwhOzu71TxjXdxuNzabzSeeb9SoUWzfvp0tW7Z4tv79+zN9+nTPf7f2ZzxRZWUl+/fvJzY21ifeQ4Bhw4adNO3D3r17adeuHeAbv29qfPjhh0RFRTFhwgTPPl94H81mM2p17ZRCo9HgdruBFvAeNnkX6PPI559/ruj1emXBggXKrl27lFtuuUUJDQ1V8vPzvR1ag1RUVCibN29WNm/erADKq6++qmzevFk5dOiQoijVwwhDQ0OVb7/9Vtm2bZsyadKkVjUU9Lbb/r+9+wtp6o3jOP4RZVsmrMgIk04QxbwIoUBJQiqUCroxjEYQCDYwI5QY5MLqql1FV0X/YGVRhBYSRUEFpjApyotcICyJXRQEXpSUFsHo+7uIRvvpRf4o1zm/9wt2MZ5n4/vlwPhw9jzPabdgMGiDg4N5W0I/f/6cm7N//35zHMcGBgZsZGTE6urqrK6uroBVz00sFrOhoSHLZDKWSqUsFotZUVGRPXz40Mzc399sft6NZeb+HqPRqA0ODlomk7Hh4WFrbGy08vJym5iYMDP392f2/diAkpISi8fjNj4+btevX7fS0lK7du1abo7bf2/Mvu/QdRzHurq6Zoy5/Tq2tLRYZWVlbut5f3+/lZeX2+HDh3NzCnkNCTu/2enTp81xHPP5fFZbW2tPnz4tdEn/2ePHj03SjFdLS4uZfd9KeOzYMVu2bJn5/X5raGiwdDpd2KLnYLbeJNnly5dzc758+WIHDhywxYsXW2lpqe3cudPevXtXuKLnqLW11VauXGk+n8+WLl1qDQ0NuaBj5v7+ZvPvsOP2HsPhsFVUVJjP57PKykoLh8N558+4vb8f7t69a2vXrjW/329VVVV28eLFvHG3/96YmT148MAkzVq326/jx48frbOz0xzHsUAgYKtWrbLu7m77+vVrbk4hr2GR2U/HGwIAAHgMa3YAAICnEXYAAICnEXYAAICnEXYAAICnEXYAAICnEXYAAICnEXYAAICnEXYAAICnEXYAAICnEXYAuNKTJ09UXFyc90BFAJgNj4sA4EqRSERlZWVKJBJKp9Navnx5oUsC8Jfizg4A15mamlJvb6/a29u1Y8cO9fT05I3fuXNHa9asUSAQ0JYtW3TlyhUVFRVpcnIyNyeZTKq+vl4LFizQihUr1NHRoenp6fltBMC8IOwAcJ2+vj5VVVUpFApp7969unTpkn7cpM5kMtq1a5eampo0OjqqtrY2dXd3533+9evX2r59u5qbm5VKpdTb26tkMqmDBw8Woh0Afxh/YwFwnY0bN2r37t3q7OxUNptVRUWFbt68qc2bNysWi+nevXt6+fJlbv7Ro0cVj8f14cMHLVq0SJFIRMXFxbpw4UJuTjKZ1KZNmzQ9Pa1AIFCItgD8IdzZAeAq6XRaz5490549eyRJJSUlCofDSiQSufGampq8z9TW1ua9Hx0dVU9Pj8rKynKvbdu26du3b8pkMvPTCIB5U1LoAgBgLhKJhLLZbN6CZDOT3+/XmTNnfuk7pqam1NbWpo6OjhljjuP8tloB/B0IOwBcI5vN6urVqzp16pS2bt2aN9bU1KQbN24oFArp/v37eWPPnz/Pe79+/XqNjY1p9erVf7xmAIXHmh0ArnH79m2Fw2FNTEwoGAzmjXV1dWlgYEB9fX0KhUI6dOiQ9u3bpxcvXigajert27eanJxUMBhUKpXShg0b1NraqkgkooULF2psbEyPHj365btDANyDNTsAXCORSKixsXFG0JGk5uZmjYyM6NOnT7p165b6+/tVXV2tc+fO5XZj+f1+SVJ1dbWGhob06tUr1dfXa926dTp+/Dhn9QAexZ0dAJ4Xj8d1/vx5vXnzptClACgA1uwA8JyzZ8+qpqZGS5Ys0fDwsE6ePMkZOsD/GGEHgOeMj4/rxIkTev/+vRzHUTQa1ZEjRwpdFoAC4W8sAADgaSxQBgAAnkbYAQAAnkbYAQAAnkbYAQAAnkbYAQAAnkbYAQAAnkbYAQAAnkbYAQAAnvYPyPcTWP2b31UAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby([\"Pclass\", \"HasCabin\"])[\"Survived\"].agg([\"mean\", \"count\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "NzvLkQEmBN76",
        "outputId": "e347a305-8474-47aa-97dd-d50912cfc136"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                     mean  count\n",
              "Pclass HasCabin                 \n",
              "1      0         0.475000     40\n",
              "       1         0.664773    176\n",
              "2      0         0.440476    168\n",
              "       1         0.812500     16\n",
              "3      0         0.235908    479\n",
              "       1         0.500000     12"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-801f2ebb-64f8-400a-bc39-df4fdefce003\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th>mean</th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Pclass</th>\n",
              "      <th>HasCabin</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th rowspan=\"2\" valign=\"top\">1</th>\n",
              "      <th>0</th>\n",
              "      <td>0.475000</td>\n",
              "      <td>40</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.664773</td>\n",
              "      <td>176</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th rowspan=\"2\" valign=\"top\">2</th>\n",
              "      <th>0</th>\n",
              "      <td>0.440476</td>\n",
              "      <td>168</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.812500</td>\n",
              "      <td>16</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th rowspan=\"2\" valign=\"top\">3</th>\n",
              "      <th>0</th>\n",
              "      <td>0.235908</td>\n",
              "      <td>479</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.500000</td>\n",
              "      <td>12</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-801f2ebb-64f8-400a-bc39-df4fdefce003')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-801f2ebb-64f8-400a-bc39-df4fdefce003 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-801f2ebb-64f8-400a-bc39-df4fdefce003');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 6,\n  \"fields\": [\n    {\n      \"column\": \"mean\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.19796975620833193,\n        \"min\": 0.2359081419624217,\n        \"max\": 0.8125,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0.475,\n          0.6647727272727273,\n          0.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 177,\n        \"min\": 12,\n        \"max\": 479,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          40,\n          176,\n          12\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for name, features in [(\"avec HasCabin\", X), (\"sans HasCabin\", X.drop(columns=[\"HasCabin\"]))]:\n",
        "    Xtr, Xte, ytr, yte = train_test_split(\n",
        "        features, y, test_size=0.2, random_state=42, stratify=y\n",
        "    )\n",
        "    model = RandomForestClassifier(random_state=42).fit(Xtr, ytr)\n",
        "    print(name, accuracy_score(yte, model.predict(Xte)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 211
        },
        "id": "Xa4L-nlSBZYp",
        "outputId": "2f066fa9-cc5c-480e-d167-7d65b4a224d8"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'X' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_777/925235826.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfor\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfeatures\u001b[0m \u001b[0;32min\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"avec HasCabin\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;34m\"sans HasCabin\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"HasCabin\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m     Xtr, Xte, ytr, yte = train_test_split(\n\u001b[1;32m      3\u001b[0m         \u001b[0mfeatures\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m42\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstratify\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m     )\n\u001b[1;32m      5\u001b[0m     \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mRandomForestClassifier\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m42\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mXtr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mytr\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'X' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df[\"Sex\"] = df[\"Sex\"].map({\"male\": 0, \"female\": 1})\n",
        "df = pd.get_dummies(df, columns=[\"Embarked\"], drop_first=True)\n",
        "df[\"FamilySize\"] = df[\"SibSp\"] + df[\"Parch\"] + 1"
      ],
      "metadata": {
        "id": "yoe6s87LBo_a"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.drop(columns=[\"Survived\", \"PassengerId\", \"Name\", \"Ticket\"])\n",
        "y = df[\"Survived\"]\n",
        "X.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "R-10wvuxBrlb",
        "outputId": "61cc6076-f92c-44f5-a84e-d5e9997db8bc"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Pclass  Sex   Age  SibSp  Parch     Fare  HasCabin  Embarked_Q  Embarked_S  \\\n",
              "0       3    0  22.0      1      0   7.2500         0       False        True   \n",
              "1       1    1  38.0      1      0  71.2833         1       False       False   \n",
              "2       3    1  26.0      0      0   7.9250         0       False        True   \n",
              "3       1    1  35.0      1      0  53.1000         1       False        True   \n",
              "4       3    0  35.0      0      0   8.0500         0       False        True   \n",
              "\n",
              "   FamilySize  \n",
              "0           2  \n",
              "1           2  \n",
              "2           1  \n",
              "3           2  \n",
              "4           1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-7d774e53-4cba-41d3-963e-6103856ee221\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "      <th>HasCabin</th>\n",
              "      <th>Embarked_Q</th>\n",
              "      <th>Embarked_S</th>\n",
              "      <th>FamilySize</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>0</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>1</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>0</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>1</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>0</td>\n",
              "      <td>False</td>\n",
              "      <td>True</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7d774e53-4cba-41d3-963e-6103856ee221')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-7d774e53-4cba-41d3-963e-6103856ee221 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-7d774e53-4cba-41d3-963e-6103856ee221');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "X",
              "summary": "{\n  \"name\": \"X\",\n  \"rows\": 891,\n  \"fields\": [\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          3,\n          1,\n          2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13.304424074688153,\n        \"min\": 0.42,\n        \"max\": 80.0,\n        \"num_unique_values\": 89,\n        \"samples\": [\n          59.0,\n          36.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 8,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 49.693428597180905,\n        \"min\": 0.0,\n        \"max\": 512.3292,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          11.2417,\n          51.8625\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"HasCabin\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked_Q\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          true,\n          false\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked_S\",\n      \"properties\": {\n        \"dtype\": \"boolean\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          false,\n          true\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"FamilySize\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 1,\n        \"max\": 11,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          8,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score"
      ],
      "metadata": {
        "id": "-koJs3DhBumd"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "for model in [LogisticRegression(max_iter=1000), RandomForestClassifier(random_state=42)]:\n",
        "    model.fit(X_train, y_train)\n",
        "    pred = model.predict(X_test)\n",
        "    print(type(model).__name__, accuracy_score(y_test, pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CNgTFhGMBx-2",
        "outputId": "fd1d27be-b94b-4992-e9ec-0641ba657e7c"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "LogisticRegression 0.8100558659217877\n",
            "RandomForestClassifier 0.776536312849162\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "models = {\n",
        "    \"Logistique\": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),\n",
        "    \"Forêt aléatoire\": RandomForestClassifier(random_state=42),\n",
        "}\n",
        "\n",
        "for name, m in models.items():\n",
        "    scores = cross_val_score(m, X, y, cv=5)\n",
        "    print(f\"{name}: {scores.mean():.3f} (± {scores.std():.3f})\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OW8KF-f1B8Ta",
        "outputId": "e3ecd762-86ab-4913-f8ac-534d7ee74e9a"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Logistique: 0.799 (± 0.012)\n",
            "Forêt aléatoire: 0.807 (± 0.033)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, classification_report\n",
        "\n",
        "model = LogisticRegression(max_iter=1000).fit(X_train, y_train)\n",
        "pred = model.predict(X_test)\n",
        "\n",
        "print(confusion_matrix(y_test, pred))\n",
        "print(classification_report(y_test, pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rLt0ca9oCA8y",
        "outputId": "f9af92de-024a-4a6b-e93f-af2e39333c8c"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[96 14]\n",
            " [20 49]]\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.83      0.87      0.85       110\n",
            "           1       0.78      0.71      0.74        69\n",
            "\n",
            "    accuracy                           0.81       179\n",
            "   macro avg       0.80      0.79      0.80       179\n",
            "weighted avg       0.81      0.81      0.81       179\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "models = {\n",
        "    \"Logistique\": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),\n",
        "    \"Forêt aléatoire\": RandomForestClassifier(random_state=42),\n",
        "}\n",
        "\n",
        "for name, m in models.items():\n",
        "    scores = cross_val_score(m, X, y, cv=5)\n",
        "    print(f\"{name}: {scores.mean():.3f} (± {scores.std():.3f})\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JoPO0-gYCMoe",
        "outputId": "9aa920b8-5618-4803-aee5-f4276bc6815a"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Logistique: 0.799 (± 0.012)\n",
            "Forêt aléatoire: 0.807 (± 0.033)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for name, features in [(\"avec HasCabin\", X), (\"sans HasCabin\", X.drop(columns=[\"HasCabin\"]))]:\n",
        "    Xtr, Xte, ytr, yte = train_test_split(\n",
        "        features, y, test_size=0.2, random_state=42, stratify=y\n",
        "    )\n",
        "    model = RandomForestClassifier(random_state=42).fit(Xtr, ytr)\n",
        "    print(name, accuracy_score(yte, model.predict(Xte)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kOflmitzCTjx",
        "outputId": "9c362c52-b64f-4c0b-89c7-7a57648be6d9"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "avec HasCabin 0.776536312849162\n",
            "sans HasCabin 0.7988826815642458\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for name, features in [(\"avec HasCabin\", X), (\"sans HasCabin\", X.drop(columns=[\"HasCabin\"]))]:\n",
        "    scores = cross_val_score(RandomForestClassifier(random_state=42), features, y, cv=5)\n",
        "    print(f\"{name}: {scores.mean():.3f} (± {scores.std():.3f})\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9BoeKoWLCc4q",
        "outputId": "eb1d639d-9b56-407f-898a-7598207bfa2b"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "avec HasCabin: 0.807 (± 0.033)\n",
            "sans HasCabin: 0.808 (± 0.032)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model = LogisticRegression(max_iter=1000, class_weight=\"balanced\").fit(X_train, y_train)\n",
        "print(classification_report(y_test, model.predict(X_test)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2xl2XKJXCgco",
        "outputId": "ddb2cf99-557d-415f-840c-d46447628d1c"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.84      0.79      0.82       110\n",
            "           1       0.70      0.77      0.73        69\n",
            "\n",
            "    accuracy                           0.78       179\n",
            "   macro avg       0.77      0.78      0.77       179\n",
            "weighted avg       0.79      0.78      0.78       179\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import classification_report\n",
        "\n",
        "# 1. Validation croisée avec / sans HasCabin\n",
        "print(\"=== Validation croisée (forêt aléatoire) ===\")\n",
        "for name, features in [(\"avec HasCabin\", X), (\"sans HasCabin\", X.drop(columns=[\"HasCabin\"]))]:\n",
        "    scores = cross_val_score(RandomForestClassifier(random_state=42), features, y, cv=5)\n",
        "    print(f\"{name}: {scores.mean():.3f} (± {scores.std():.3f})\")\n",
        "\n",
        "# 2. Régression logistique avec class_weight=\"balanced\"\n",
        "print(\"\\n=== Logistique, class_weight='balanced' ===\")\n",
        "model = LogisticRegression(max_iter=1000, class_weight=\"balanced\").fit(X_train, y_train)\n",
        "print(classification_report(y_test, model.predict(X_test)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yNBFzBR3DRD2",
        "outputId": "3cf48812-a654-470a-dbee-8ce7f4a74157"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "=== Validation croisée (forêt aléatoire) ===\n",
            "avec HasCabin: 0.807 (± 0.033)\n",
            "sans HasCabin: 0.808 (± 0.032)\n",
            "\n",
            "=== Logistique, class_weight='balanced' ===\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.84      0.79      0.82       110\n",
            "           1       0.70      0.77      0.73        69\n",
            "\n",
            "    accuracy                           0.78       179\n",
            "   macro avg       0.77      0.78      0.77       179\n",
            "weighted avg       0.79      0.78      0.78       179\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import HistGradientBoostingClassifier\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "X2 = X.drop(columns=[\"HasCabin\"])\n",
        "\n",
        "models = {\n",
        "    \"Logistique\": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),\n",
        "    \"Forêt aléatoire\": RandomForestClassifier(random_state=42),\n",
        "    \"Gradient boosting\": HistGradientBoostingClassifier(random_state=42),\n",
        "}\n",
        "\n",
        "for name, m in models.items():\n",
        "    scores = cross_val_score(m, X2, y, cv=5)\n",
        "    print(f\"{name}: {scores.mean():.3f} (± {scores.std():.3f})\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yrfHekczDcN2",
        "outputId": "140c7ca1-deb8-4bc3-9b2e-f24f27ea907c"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Logistique: 0.799 (± 0.015)\n",
            "Forêt aléatoire: 0.808 (± 0.032)\n",
            "Gradient boosting: 0.829 (± 0.028)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "param_grid = {\n",
        "    \"n_estimators\": [100, 300],\n",
        "    \"max_depth\": [3, 5, 8, None],\n",
        "    \"min_samples_leaf\": [1, 3, 5],\n",
        "}\n",
        "\n",
        "grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)\n",
        "grid.fit(X2, y)\n",
        "\n",
        "print(\"Meilleurs paramètres :\", grid.best_params_)\n",
        "print(f\"Meilleur score CV : {grid.best_score_:.3f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bO3bvGyxDhzN",
        "outputId": "84c11831-cd7f-43ea-9ef1-edc5f05e13f6"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Meilleurs paramètres : {'max_depth': 8, 'min_samples_leaf': 1, 'n_estimators': 100}\n",
            "Meilleur score CV : 0.829\n"
          ]
        }
      ]
    }
  ]
}
