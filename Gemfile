# frozen_string_literal: true

source "https://rubygems.org"

gemspec

gem 'tzinfo'
gem 'tzinfo-data'

gem 'csv'
gem 'base64'

group :test do
  gem "html-proofer", "~> 5.0"
end

# Windows 환경에서만 wdm 설치
install_if -> { Gem.win_platform? } do
  gem 'wdm', '>= 0.1.0'
end