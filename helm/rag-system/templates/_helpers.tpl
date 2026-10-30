{{/*
RAG System Common Labels
*/}}
{{- define "rag.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
RAG System Selector Labels
*/}}
{{- define "rag.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Full name
*/}}
{{- define "rag.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Backend service name
*/}}
{{- define "rag.backendServiceName" -}}
{{- printf "%s-backend" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Frontend service name
*/}}
{{- define "rag.frontendServiceName" -}}
{{- printf "%s-frontend" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
vLLM service name
*/}}
{{- define "rag.vllmServiceName" -}}
{{- printf "%s-vllm" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ChromaDB service name
*/}}
{{- define "rag.chromadbServiceName" -}}
{{- printf "%s-chromadb" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
