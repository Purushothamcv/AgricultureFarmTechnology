# Docker Deployment Guide - SmartAgri-AI

This guide provides comprehensive instructions for deploying the SmartAgri-AI application using Docker across multiple cloud platforms.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Environment Variables](#environment-variables)
4. [Platform-Specific Deployments](#platform-specific-deployments)
   - [Render](#deploy-to-render)
   - [Railway](#deploy-to-railway)
   - [AWS EC2](#deploy-to-aws-ec2)
   - [DigitalOcean](#deploy-to-digitalocean)
   - [Google Cloud Platform (GCP)](#deploy-to-google-cloud-platform)
5. [Production Considerations](#production-considerations)

---

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- At least 2GB RAM available
- 10GB free disk space

---

## Local Development

### 1. Clone the Repository
```bash
git clone https://github.com/Purushothamcv/AgricultureFarmTechnology.git
cd SmartAgri-AI
```

### 2. Set Up Environment Variables

Create `.env` files in both backend and frontend directories:

**Backend `.env`:**
```env
MONGODB_URL=mongodb://mongodb:27017/FinalProject
SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=10080

GOOGLE_CLIENT_ID=745305741156-di4f6tc9o7p6773hp21mh60u16m3anik.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret

GROQ_API_KEY=your-groq-api-key

OPENWEATHER_API_KEY=your-openweather-api-key
NEWSAPI_KEY=your-newsapi-key
```

**Frontend `.env`:**
```env
VITE_API_BASE_URL=http://localhost:8001
VITE_GOOGLE_CLIENT_ID=745305741156-di4f6tc9o7p6773hp21mh60u16m3anik.apps.googleusercontent.com
```

### 3. Build and Run with Docker Compose

```bash
# Build all services
docker-compose build

# Start all services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes (cleans MongoDB data)
docker-compose down -v
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- API Documentation: http://localhost:8001/docs
- MongoDB: localhost:27017 (internal only)

---

## Environment Variables

### Required Backend Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGODB_URL` | MongoDB connection string | `mongodb://mongodb:27017/FinalProject` |
| `SECRET_KEY` | JWT secret key (use strong random string) | `openssl rand -hex 32` |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | `745305741156-...` |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret | Get from Google Console |
| `GROQ_API_KEY` | Groq API key for chatbot | Get from Groq |

### Optional Backend Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Backend port | `8000` |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` |
| `JWT_EXPIRATION_MINUTES` | Token expiration | `10080` (7 days) |
| `OPENWEATHER_API_KEY` | Weather data | Required for weather features |
| `NEWSAPI_KEY` | News data | Required for news features |

### Required Frontend Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `https://your-backend-url.com` |
| `VITE_GOOGLE_CLIENT_ID` | Google OAuth client ID | Same as backend |

---

## Platform-Specific Deployments

## Deploy to Render

### Backend Deployment

1. **Create a New Web Service**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   ```yaml
   Name: smartagri-backend
   Environment: Docker
   Region: Choose nearest to your users
   Branch: main
   Dockerfile Path: ./backend/Dockerfile
   ```

3. **Set Environment Variables**
   - Go to "Environment" tab
   - Add all required backend variables
   - Update `MONGODB_URL` to your MongoDB Atlas connection string

4. **Advanced Settings**
   ```yaml
   Docker Command: uvicorn main_fastapi:app --host 0.0.0.0 --port $PORT --workers 4
   Health Check Path: /health
   ```

### Frontend Deployment

1. **Create a New Static Site**
   - Click "New +" → "Static Site"
   - Connect your repository

2. **Configure Service**
   ```yaml
   Name: smartagri-frontend
   Build Command: cd frontend && npm install && npm run build
   Publish Directory: frontend/dist
   ```

3. **Environment Variables**
   ```env
   VITE_API_BASE_URL=https://your-backend-service.onrender.com
   VITE_GOOGLE_CLIENT_ID=745305741156-di4f6tc9o7p6773hp21mh60u16m3anik.apps.googleusercontent.com
   ```

### MongoDB on Render

Use [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (free tier available):
1. Create a cluster
2. Whitelist Render's IP addresses (0.0.0.0/0 for simplicity)
3. Get connection string and add to backend environment variables

---

## Deploy to Railway

### Using Railway CLI

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Initialize Project**
   ```bash
   railway init
   railway link
   ```

3. **Deploy Services**
   ```bash
   # Deploy MongoDB
   railway add --plugin mongodb
   
   # Deploy backend
   cd backend
   railway up --service backend
   
   # Deploy frontend
   cd ../frontend
   railway up --service frontend
   ```

### Using Railway Dashboard

1. **Create New Project**
   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project" → "Deploy from GitHub repo"

2. **Add MongoDB Service**
   - Click "Add Service" → "Database" → "MongoDB"
   - Copy the connection URL

3. **Add Backend Service**
   - Click "Add Service" → "GitHub Repo" → Select your repo
   - Root Directory: `/backend`
   - Add environment variables (use MongoDB connection from step 2)

4. **Add Frontend Service**
   - Click "Add Service" → "GitHub Repo" → Select your repo
   - Root Directory: `/frontend`
   - Build Command: `npm install && npm run build`
   - Start Command: `npx serve -s dist -l $PORT`
   - Add `VITE_API_BASE_URL` pointing to backend service URL

5. **Configure Networking**
   - Backend: Generate domain (e.g., `smartagri-api.railway.app`)
   - Frontend: Generate domain (e.g., `smartagri.railway.app`)
   - Update frontend's `VITE_API_BASE_URL`

---

## Deploy to AWS EC2

### Prerequisites
- AWS Account
- EC2 instance (t2.medium or higher recommended)
- Security group with ports 80, 443, 22 open

### 1. Launch EC2 Instance

```bash
# Connect to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Clone and Configure Application

```bash
# Clone repository
git clone https://github.com/Purushothamcv/AgricultureFarmTechnology.git
cd SmartAgri-AI

# Create environment files
nano backend/.env   # Add your environment variables
nano frontend/.env  # Add your environment variables
```

### 3. Update docker-compose.yml for Production

```yaml
# Update frontend ports to expose on 80
services:
  frontend:
    ports:
      - "80:80"
```

### 4. Deploy Application

```bash
# Build and start services
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f

# Check running containers
docker ps
```

### 5. Set Up SSL with Let's Encrypt (Optional but Recommended)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Stop frontend container temporarily
docker-compose stop frontend

# Get SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Update nginx configuration to use SSL
# Add certificate paths to frontend/nginx.conf

# Restart frontend
docker-compose up -d frontend
```

### 6. Auto-restart on Reboot

```bash
# Create systemd service
sudo nano /etc/systemd/system/smartagri.service
```

```ini
[Unit]
Description=SmartAgri Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/SmartAgri-AI
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable smartagri
sudo systemctl start smartagri
```

---

## Deploy to DigitalOcean

### Option 1: DigitalOcean App Platform (Easiest)

1. **Create New App**
   - Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
   - Click "Create App" → Select GitHub repo

2. **Configure Backend**
   ```yaml
   Name: smartagri-backend
   Type: Web Service
   Dockerfile Path: backend/Dockerfile
   HTTP Port: 8000
   Environment Variables: [Add all required variables]
   ```

3. **Configure Frontend**
   ```yaml
   Name: smartagri-frontend
   Type: Static Site
   Build Command: cd frontend && npm install && npm run build
   Output Directory: frontend/dist
   Environment Variables:
     VITE_API_BASE_URL: ${smartagri-backend.PUBLIC_URL}
   ```

4. **Add MongoDB Database**
   - Click "Create Resource" → "Database" → "MongoDB"
   - Update backend `MONGODB_URL` with connection string

### Option 2: DigitalOcean Droplet (More Control)

1. **Create Droplet**
   - Choose Ubuntu 22.04 LTS
   - Basic plan: 2GB RAM / 1 CPU ($12/month)
   - Select datacenter region
   - Add SSH key

2. **Follow AWS EC2 deployment steps** (they're the same for Ubuntu)

---

## Deploy to Google Cloud Platform

### Option 1: Cloud Run (Serverless, Recommended)

#### Deploy Backend

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
gcloud init

# Build and push backend image
cd backend
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/smartagri-backend

# Deploy to Cloud Run
gcloud run deploy smartagri-backend \
  --image gcr.io/YOUR_PROJECT_ID/smartagri-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MONGODB_URL="your-mongodb-url",SECRET_KEY="your-secret" \
  --memory 2Gi \
  --timeout 300
```

#### Deploy Frontend

```bash
# Build frontend
cd frontend
npm install
npm run build

# Deploy to Cloud Storage + Cloud CDN
gsutil mb gs://YOUR_BUCKET_NAME
gsutil -m cp -r dist/* gs://YOUR_BUCKET_NAME
gsutil iam ch allUsers:objectViewer gs://YOUR_BUCKET_NAME

# Make bucket a website
gsutil web set -m index.html -e index.html gs://YOUR_BUCKET_NAME
```

### Option 2: Google Kubernetes Engine (GKE)

```bash
# Create GKE cluster
gcloud container clusters create smartagri-cluster \
  --num-nodes=3 \
  --machine-type=e2-medium \
  --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials smartagri-cluster --zone=us-central1-a

# Create Kubernetes deployment files (see k8s/ directory if needed)

# Apply configurations
kubectl apply -f k8s/mongodb.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml

# Expose frontend with LoadBalancer
kubectl expose deployment frontend --type=LoadBalancer --port=80
```

### Option 3: Compute Engine VM

1. **Create VM Instance**
   ```bash
   gcloud compute instances create smartagri-vm \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud \
     --machine-type=e2-medium \
     --zone=us-central1-a \
     --tags=http-server,https-server
   ```

2. **SSH into VM and follow AWS EC2 deployment steps**

---

## Production Considerations

### Security

1. **Environment Variables**
   - Never commit `.env` files
   - Use strong, unique `SECRET_KEY` for JWT
   - Rotate credentials regularly
   - Use secrets management (AWS Secrets Manager, GCP Secret Manager)

2. **Database**
   - Enable MongoDB authentication
   - Use connection string with username/password
   - Enable SSL/TLS connections
   - Regular backups (automated snapshots)
   - Whitelist only necessary IP addresses

3. **API Security**
   - Enable CORS only for your frontend domain
   - Implement rate limiting
   - Use HTTPS everywhere (Let's Encrypt for free SSL)
   - Keep dependencies updated

4. **Docker Security**
   - Don't run containers as root
   - Scan images for vulnerabilities: `docker scan`
   - Use specific image tags, not `latest`
   - Minimize image layers

### Performance

1. **Backend Optimization**
   - Use production ASGI server (uvicorn with multiple workers)
   - Enable gzip compression
   - Implement caching (Redis) for frequent queries
   - Database indexing for query optimization

2. **Frontend Optimization**
   - Enable nginx gzip compression (already configured)
   - Use CDN for static assets
   - Implement lazy loading for routes
   - Minimize bundle size

3. **Monitoring**
   - Set up application monitoring (Sentry, Datadog)
   - Log aggregation (ELK stack, CloudWatch)
   - Health check endpoints
   - Uptime monitoring (UptimeRobot, Pingdom)

### Scaling

1. **Horizontal Scaling**
   ```bash
   # Scale backend containers
   docker-compose up -d --scale backend=3
   ```

2. **Load Balancing**
   - Add nginx/traefik as reverse proxy
   - Use cloud load balancers (ALB, GCP Load Balancer)
   - Session affinity for stateful operations

3. **Database Scaling**
   - MongoDB replica sets for high availability
   - Read replicas for read-heavy workloads
   - Sharding for large datasets

### Backup Strategy

1. **Database Backups**
   ```bash
   # Automated MongoDB backup
   docker exec mongodb mongodump --out=/backup/$(date +%Y%m%d)
   ```

2. **Application State**
   - Version control for code (Git)
   - Container registry for images
   - Backup ML models to cloud storage

### Maintenance

1. **Updates**
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Rebuild and restart
   docker-compose build
   docker-compose up -d
   ```

2. **Log Management**
   ```bash
   # View logs
   docker-compose logs -f --tail=100
   
   # Rotate logs to prevent disk full
   docker-compose logs --no-log-prefix > logs_$(date +%Y%m%d).txt
   ```

3. **Cleanup**
   ```bash
   # Remove unused images
   docker image prune -a
   
   # Remove unused volumes
   docker volume prune
   ```

---

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   sudo lsof -i :8001
   
   # Kill process
   sudo kill -9 PID
   ```

2. **MongoDB Connection Failed**
   - Check `MONGODB_URL` environment variable
   - Verify MongoDB container is running: `docker ps`
   - Check logs: `docker-compose logs mongodb`

3. **Frontend Can't Connect to Backend**
   - Verify `VITE_API_BASE_URL` is correct
   - Check CORS settings in backend
   - Ensure backend health endpoint works: `curl http://backend:8000/health`

4. **ML Models Not Loading**
   - Verify model files exist in `backend/model/` directory
   - Check file permissions
   - Review backend logs for loading errors

5. **Build Failures**
   - Clear Docker cache: `docker-compose build --no-cache`
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt/package.json

---

## Support

For issues and questions:
- GitHub Issues: [Repository Issues](https://github.com/Purushothamcv/AgricultureFarmTechnology/issues)
- Documentation: See individual service README files

---

## License

[Add your license information here]
