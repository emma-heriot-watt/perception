# Changelog

All notable changes to this project will be documented in this file. See
[Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## [1.10.1](https://github.com/emma-simbot/perception/compare/v1.10.0...v1.10.1) (2023-02-17)


### Bug Fixes

* place entity classifier on correct device ([#226](https://github.com/emma-simbot/perception/issues/226)) ([5f16fb5](https://github.com/emma-simbot/perception/commit/5f16fb538d36ec7918efe2053cda61be3106f2b0))

## [1.10.0](https://github.com/emma-simbot/perception/compare/v1.9.0...v1.10.0) (2023-02-15)


### Features

* optimised perception using torch response ([#224](https://github.com/emma-simbot/perception/issues/224)) ([831c680](https://github.com/emma-simbot/perception/commit/831c6800a8210dfa64ef53d00741c28d912ffbfb))

## [1.9.0](https://github.com/emma-simbot/perception/compare/v1.8.2...v1.9.0) (2023-02-14)


### Features

* Include object entities ([#223](https://github.com/emma-simbot/perception/issues/223)) ([c0b9d97](https://github.com/emma-simbot/perception/commit/c0b9d974c9acd0a97e71c7b643e920df9dac50bf))


### Bug Fixes

* change endpoints to be non-asynchronous ([#221](https://github.com/emma-simbot/perception/issues/221)) ([57f9668](https://github.com/emma-simbot/perception/commit/57f9668aaa899d5932ff13f3cab629cbc701d220))

## [1.8.2](https://github.com/emma-simbot/perception/compare/v1.8.1...v1.8.2) (2023-01-06)


### Bug Fixes

* pin numpy to `<1.24` because deprecations break legacy code ([dfe3b47](https://github.com/emma-simbot/perception/commit/dfe3b47dda8518c3ab9376aba732be8fda6af3f0)), closes [#208](https://github.com/emma-simbot/perception/issues/208)

## [1.8.1](https://github.com/emma-simbot/perception/compare/v1.8.0...v1.8.1) (2023-01-05)


### Bug Fixes

* Change response_model to response_class ([#206](https://github.com/emma-simbot/perception/issues/206)) ([75119d8](https://github.com/emma-simbot/perception/commit/75119d89277ad72d5527801f6110e4c074db0438))

## [1.8.0](https://github.com/emma-simbot/perception/compare/v1.7.0...v1.8.0) (2023-01-02)


### Features

* **telemetry:** include instrumentation and tracking of metrics for requests and the system ([8ae3643](https://github.com/emma-simbot/perception/commit/8ae3643db9bcb85d3c2ca4b853094da0487a4a5b))
* **telemetry:** More fine-grained tracing during feature extraction ([0f2b513](https://github.com/emma-simbot/perception/commit/0f2b513a3c045db0c2f35c4493067d3815eb9c1d))
* use orjson to build response and add tracing ([435e89b](https://github.com/emma-simbot/perception/commit/435e89b47cd40ed8883eb070c434d87c64d365d2))


### Bug Fixes

* fastapi ORJSONResponse import ([ab3fe3d](https://github.com/emma-simbot/perception/commit/ab3fe3d912e74299f2336395851f06fcd4d8b9df))

## [1.7.0](https://github.com/emma-simbot/perception/compare/v1.6.1...v1.7.0) (2022-12-23)


### Features

* added support for tracing ([#203](https://github.com/emma-simbot/perception/issues/203)) ([7529955](https://github.com/emma-simbot/perception/commit/7529955c3e39101c1618f1499ea5238ebd98d61f))

## [1.6.1](https://github.com/emma-simbot/perception/compare/v1.6.0...v1.6.1) (2022-12-22)


### Bug Fixes

* **revert:** using Ray Serve to serve Perception ([#200](https://github.com/emma-simbot/perception/issues/200)) ([#202](https://github.com/emma-simbot/perception/issues/202)) ([3091da6](https://github.com/emma-simbot/perception/commit/3091da6a25ca86cf66281884e73757ab9adb2445))

## [1.6.0](https://github.com/emma-simbot/perception/compare/v1.5.1...v1.6.0) (2022-12-21)


### Features

* Using Ray Serve to serve Perception ([#200](https://github.com/emma-simbot/perception/issues/200)) ([e83a960](https://github.com/emma-simbot/perception/commit/e83a9602badbf60df1fc7741a668d2ca34508f3b))

## [1.5.1](https://github.com/emma-simbot/perception/compare/v1.5.0...v1.5.1) (2022-12-20)


### Bug Fixes

* revert adding tracing for the feature extractor endpoint ([#198](https://github.com/emma-simbot/perception/issues/198)) ([#199](https://github.com/emma-simbot/perception/issues/199)) ([dcaa484](https://github.com/emma-simbot/perception/commit/dcaa484ac233130f8c809846d00296160e7491ea))

## [1.5.0](https://github.com/emma-simbot/perception/compare/v1.4.0...v1.5.0) (2022-12-20)


### Features

* setup tracing for the feature extractor endpoint ([#198](https://github.com/emma-simbot/perception/issues/198)) ([b86065d](https://github.com/emma-simbot/perception/commit/b86065d41435cb943fd14f2e423eabf9ac44482e))

## [1.4.0](https://github.com/emma-simbot/perception/compare/v1.3.0...v1.4.0) (2022-11-28)


### Features

* Add simbot custom model config ([#191](https://github.com/emma-simbot/perception/issues/191)) ([1793f33](https://github.com/emma-simbot/perception/commit/1793f33a10c65d542b01f416b2dbfdfb87847ba5))

## [1.3.0](https://github.com/emma-simbot/perception/compare/v1.2.0...v1.3.0) (2022-11-28)


### Features

* use custom classmap ([#189](https://github.com/emma-simbot/perception/issues/189)) ([864a274](https://github.com/emma-simbot/perception/commit/864a2749cbb168f96681d8790edbce424b0f0269))

## [1.2.0](https://github.com/emma-simbot/perception/compare/v1.1.0...v1.2.0) (2022-11-19)


### Features

* Change simbot classmap path ([#187](https://github.com/emma-simbot/perception/issues/187)) ([66dd978](https://github.com/emma-simbot/perception/commit/66dd9788b67b1f1af372a90973ff1e67c327f0b6))

## [1.1.0](https://github.com/emma-simbot/perception/compare/v1.0.0...v1.1.0) (2022-11-04)


### Features

* Update simbot classmap config ([#185](https://github.com/emma-simbot/perception/issues/185)) ([4fbcca7](https://github.com/emma-simbot/perception/commit/4fbcca7dc89f3efccd9104748f624ed4d486a72c))

## 1.0.0 (2022-10-28)


### Bug Fixes

* use the request body for updating the model device ([8f83fc6](https://github.com/emma-simbot/perception/commit/8f83fc6567b9a77f739826d14e2f6856deebddfe))


### Reverts

* Revert "Build(deps): bump actions/setup-python from 3 to 4 (#152)" (#162) ([ee64b1c](https://github.com/emma-simbot/perception/commit/ee64b1c14e8657e1d0ceb0d1314e2e0db017bbc7)), closes [#152](https://github.com/emma-simbot/perception/issues/152) [#162](https://github.com/emma-simbot/perception/issues/162)
