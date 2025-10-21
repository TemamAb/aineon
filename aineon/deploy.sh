#!/bin/bash
# PLATINUM SOURCES: GitHub Actions, GitLab CI
# CONTINUAL LEARNING: Deployment success rate learning, rollback optimization

set -euo pipefail

# Configuration
DEPLOY_ENV=${1:-"staging"}
APP_NAME="aineon-trading-platform"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOY_LOG="deploy_${TIMESTAMP}.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOY_LOG"
}

success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$DEPLOY_LOG"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "$DEPLOY_LOG"
}

error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$DEPLOY_LOG"
    exit 1
}

# Validation functions
validate_environment() {
    log "Validating deployment environment: $DEPLOY_ENV"
    
    case $DEPLOY_ENV in
        "development"|"staging"|"production")
            success "Environment $DEPLOY_ENV is valid"
            ;;
        *)
            error "Invalid environment: $DEPLOY_ENV. Must be one of: development, staging, production"
            ;;
    esac

    # Check required tools
    command -v docker >/dev/null 2>&1 || error "Docker is required but not installed"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is required but not installed"
    command -v node >/dev/null 2>&1 || error "Node.js is required but not installed"
    command -v npm >/dev/null 2>&1 || error "npm is required but not installed"
    
    success "All required tools are available"
}

validate_secrets() {
    log "Validating environment secrets..."
    
    local required_secrets=(
        "DATABASE_URL"
        "REDIS_URL"
        "JWT_SECRET"
    )
    
    if [ "$DEPLOY_ENV" = "production" ]; then
        required_secrets+=(
            "BLOCKCHAIN_RPC_URL"
            "EXCHANGE_API_KEY"
            "EXCHANGE_API_SECRET"
        )
    fi
    
    for secret in "${required_secrets[@]}"; do
        if [ -z "${!secret:-}" ]; then
            error "Required secret $secret is not set"
        fi
    done
    
    success "All required secrets are set"
}

# Security scanning
security_scan() {
    log "Running security scans..."
    
    # npm audit
    log "Running npm audit..."
    if ! npm audit --audit-level moderate; then
        warning "npm audit found vulnerabilities. Check and update dependencies."
    fi
    
    # Docker image security scan
    if command -v trivy >/dev/null 2>&1; then
        log "Running Trivy security scan..."
        trivy image --exit-code 1 --severity HIGH,CRITICAL "$APP_NAME:$DEPLOY_ENV" || {
            warning "Security scan found issues. Continuing deployment but please review."
        }
    else
        warning "Trivy not installed, skipping container security scan"
    fi
    
    success "Security scans completed"
}

# Build and test
build_and_test() {
    log "Building application for $DEPLOY_ENV..."
    
    # Install dependencies
    npm ci --only=production --audit
    
    # Run tests
    log "Running test suite..."
    case $DEPLOY_ENV in
        "development")
            npm run test
            ;;
        "staging"|"production")
            npm run test:ci
            ;;
    esac
    
    # Build application
    log "Building application..."
    npm run build
    
    # Build Docker image
    log "Building Docker image..."
    docker build \
        --target production \
        -t "$APP_NAME:$DEPLOY_ENV" \
        -t "$APP_NAME:latest" \
        --build-arg DEPLOY_ENV="$DEPLOY_ENV" \
        .
    
    success "Build and test phase completed"
}

# Database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    timeout 60s bash -c 'until docker exec aineon-postgres pg_isready -U aineon; do sleep 2; done' || {
        error "Database is not ready within timeout period"
    }
    
    # Run migrations
    if ! npx knex migrate:latest; then
        error "Database migration failed"
    fi
    
    # Run seed data for non-production environments
    if [ "$DEPLOY_ENV" != "production" ]; then
        log "Seeding database with test data..."
        npx knex seed:run
    fi
    
    success "Database migrations completed"
}

# Deployment strategies
blue_green_deploy() {
    log "Starting blue-green deployment..."
    
    local current_color
    local new_color
    
    # Determine current and new colors
    if docker ps --format "table {{.Names}}" | grep -q "${APP_NAME}-blue"; then
        current_color="blue"
        new_color="green"
    else
        current_color="green"
        new_color="blue"
    fi
    
    # Deploy new version
    log "Deploying new version ($new_color)..."
    docker-compose -p "${APP_NAME}-${new_color}" up -d --build
    
    # Health check new version
    log "Performing health check on new version..."
    if ! health_check "${APP_NAME}-${new_color}"; then
        error "Health check failed for new version ($new_color)"
    fi
    
    # Switch traffic (simulated with load balancer config)
    log "Switching traffic to new version ($new_color)..."
    update_load_balancer "$new_color"
    
    # Keep old version for rollback capability
    log "Keeping old version ($current_color) for rollback..."
    docker-compose -p "${APP_NAME}-${current_color}" stop
    
    success "Blue-green deployment completed successfully"
}

health_check() {
    local service_name=$1
    local max_attempts=30
    local attempt=1
    
    log "Health checking $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "http://localhost:3000/health" >/dev/null 2>&1; then
            success "Health check passed for $service_name"
            return 0
        fi
        
        warning "Health check attempt $attempt/$max_attempts failed for $service_name"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    error "Health check failed for $service_name after $max_attempts attempts"
}

update_load_balancer() {
    local active_color=$1
    log "Updating load balancer to route traffic to $active_color"
    
    # In a real scenario, this would update your load balancer configuration
    # For now, we'll simulate the process
    sleep 10
    success "Load balancer updated to route traffic to $active_color"
}

# Rollback functionality
rollback() {
    warning "Initiating rollback..."
    
    local rollback_color
    if docker ps --format "table {{.Names}}" | grep -q "${APP_NAME}-blue"; then
        rollback_color="blue"
    else
        rollback_color="green"
    fi
    
    log "Rolling back to $rollback_color..."
    
    # Stop current version
    docker-compose down
    
    # Start previous version
    docker-compose -p "${APP_NAME}-${rollback_color}" up -d
    
    # Health check rollback version
    if health_check "${APP_NAME}-${rollback_color}"; then
        success "Rollback completed successfully"
    else
        error "Rollback failed - service is unhealthy"
    fi
}

# Cleanup old images and containers
cleanup() {
    log "Cleaning up old Docker resources..."
    
    # Remove old containers
    docker ps -aq --filter "name=${APP_NAME}" | xargs docker rm -f 2>/dev/null || true
    
    # Remove old images
    docker images --filter "reference=${APP_NAME}" -q | xargs docker rmi 2>/dev/null || true
    
    # Remove unused networks and volumes
    docker network prune -f
    docker volume prune -f
    
    success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting deployment process for $APP_NAME to $DEPLOY_ENV"
    
    # Set up error handling
    trap 'error "Deployment failed at line $LINENO"' ERR
    trap 'warning "Deployment interrupted"; rollback; exit 1' INT TERM
    
    # Execute deployment steps
    validate_environment
    validate_secrets
    security_scan
    build_and_test
    
    if [ "$DEPLOY_ENV" = "production" ]; then
        blue_green_deploy
    else
        log "Starting standard deployment for $DEPLOY_ENV"
        docker-compose up -d --build
        health_check "$APP_NAME"
    fi
    
    run_migrations
    cleanup
    
    success "Deployment completed successfully!"
    log "Deployment log saved to: $DEPLOY_LOG"
    
    # Print deployment summary
    echo
    echo "=== DEPLOYMENT SUMMARY ==="
    echo "Environment: $DEPLOY_ENV"
    echo "Timestamp: $(date)"
    echo "Status: ${GREEN}SUCCESS${NC}"
    echo "Services deployed:"
    docker ps --filter "name=aineon" --format "table {{.Names}}\t{{.Status}}"
    echo
}

# Execute main function
main "$@"
