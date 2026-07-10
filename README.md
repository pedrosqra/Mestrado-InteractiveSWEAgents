# Reprodução e Extensão da RQ3 do Ambig-SWE (PPGCC/UFCG)

Este repositório é um fork de [InteractiveSWEAgents](https://github.com/sani903/InteractiveSWEAgents), o benchmark **Ambig-SWE** construído sobre o framework [OpenHands](https://github.com/All-Hands-AI/OpenHands). Ele estende o projeto original com a reprodução parcial e a extensão da RQ3 (qualidade das perguntas de clarificação), aplicada à família open-weight Qwen2.5 Coder (1.5B, 7B, 14B e 32B) sob dois simuladores de usuário (GPT-4o-mini e Gemini 3.5 Flash).

- Guia completo de reprodução, dados processados e scripts: [reproducao/README.md](reproducao/README.md)
- Suíte de validação estatística: [reproducao/validation/](reproducao/validation/)
- Logs brutos das 240 execuções (Zenodo): [DOI 10.5281/zenodo.21211716](https://doi.org/10.5281/zenodo.21211716)

Toda a documentação, o código de análise e os artefatos da reprodução ficam na pasta `reproducao/`. O restante do repositório preserva a base do OpenHands/Ambig-SWE para permitir a execução idêntica à do estudo original; para a documentação do framework, consulte os repositórios linkados acima.

## Licença

Distribuído sob a Licença MIT, herdada do projeto original. Veja [`LICENSE`](./LICENSE).

## Citação (trabalho original)

Esta é uma reprodução e extensão do Ambig-SWE. Ao utilizá-la, cite o paper original:

```
@misc{vijayvargiya2025interactiveagentsovercomeambiguity,
      title={Interactive Agents to Overcome Ambiguity in Software Engineering},
      author={Sanidhya Vijayvargiya and Xuhui Zhou and Akhila Yerukola and Maarten Sap and Graham Neubig},
      year={2025},
      eprint={2502.13069},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.13069},
}
```
