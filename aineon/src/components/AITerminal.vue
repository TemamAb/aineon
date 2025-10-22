<template>
  <div class="component-placeholder">
    <h3>{{ componentName }} - WORKING</h3>
    <p>Component loaded successfully</p>
  </div>
</template>

<script>
export default {
  name: 'ComponentPlaceholder',
  data() {
    return {
      componentName: this.$options.name
    }
  }
}
</script>

<style scoped>
.component-placeholder {
  padding: 20px;
  border: 1px solid #333;
  border-radius: 8px;
  background: #1a1a1a;
  text-align: center;
}
</style>
