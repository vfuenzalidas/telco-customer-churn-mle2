# Estrategia Git del proyecto

## Objetivo

Este documento describe la estrategia de ramas, commits, Pull Requests y versiones utilizada en el proyecto de predicción de abandono de clientes.

## Ramas principales

### `main`

Contiene las versiones estables y entregables del proyecto.

Solamente recibe cambios aprobados desde `development` mediante Pull Request. Las versiones oficiales se identifican mediante tags y releases.

### `development`

Contiene la integración de los cambios que serán incorporados en la próxima versión estable.

Recibe cambios desde ramas `feature/*` mediante Pull Request.

## Ramas de trabajo

Cada mejora se desarrolla en una rama independiente creada desde `development`.

Convención utilizada:

```text
feature/nombre-del-cambio