{
  "type" : "index_hadoop",
  "spec" : {
    "ioConfig" : {
      "type" : "hadoop",
      "inputSpec" : {
        "type" : "static",
        "paths" : "Movto_201701010000_201701010100.csv"
      }
    },
    "dataSchema" : {
      "dataSource" : "sptrans",
      "granularitySpec" : {
        "type" : "uniform",
        "segmentGranularity" : "day",
        "queryGranularity" : "day",
        "intervals" : ["2017-01-01/2017-01-02"]
      },
      "parser" : {
        "type" : "hadoopyString",
        "parseSpec" : {
          "format" : "csv",
          "columns" : ["cd_evento_avl_movto","dt_movto","nr_identificador","nr_evento_linha","nr_ponto","nr_velocidade","nr_voltagem","nr_temperatura_interna","nr_evento_terminal_dado","nr_evento_es_1","nr_latitude_grau","nr_longitude_grau","nr_indiceregistro","dt_avl","nr_distancia", "cd_tipo_veiculo_geo", "cd_avl_conexao", "cd_prefixo", "CodigoLinha", "Circular", "Letreiro", "Sentido", "Tipo", "DenominacaoTPTS", "DenominacaoTSTP", "Informacoes"],
          "dimensionsSpec" : {
            "dimensions" : ["cd_evento_avl_movto","dt_movto","nr_identificador","nr_evento_linha","nr_ponto","nr_evento_terminal_dado","nr_evento_es_1","nr_latitude_grau","nr_longitude_grau","nr_indiceregistro", "cd_tipo_veiculo_geo", "cd_avl_conexao", "cd_prefixo", "CodigoLinha", "Circular", "Letreiro", "Sentido", "Tipo", "DenominacaoTPTS", "DenominacaoTSTP", "Informacoes"]
          },
          "timestampSpec" : {
            "format" : "auto",
            "column" : "dt_avl"
          }
        }
      },
      "metricsSpec" : [
        {
          "name": "nr_velocidade",
          "type": "longSum",
          "fieldName": "nr_velocidade"
        },
        {
          "name": "nr_voltagem",
          "type": "longSum",
          "fieldName": "nr_voltagem"
        },
        {
          "name": "nr_temperatura_interna",
          "type": "longSum",
          "fieldName": "nr_temperatura_interna"
        },
        {
          "name": "nr_distancia",
          "type": "longSum",
          "fieldName": "nr_distancia"
        }
      ]
    },
    "tuningConfig" : {
      "type" : "hadoop",
      "partitionsSpec" : {
        "type" : "hashed",
        "targetPartitionSize" : 5000000
      },
      "jobProperties" : {}
    }
  }
}